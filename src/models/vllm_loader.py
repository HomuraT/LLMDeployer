import logging
import os
import signal
import subprocess
import socket
import time

import psutil
import requests
from huggingface_hub import login
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
from src.utils.process_utils import get_pid_by_grep
from src.utils.yaml_utils import YAMLConfigManager
logging.basicConfig(level=logging.INFO)

def load_model(model_name:str, vllm_config=None)->LLM:
    if vllm_config is None:
        vllm_config = {}
    config = YAMLConfigManager.read_yaml(os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name+'.yaml'))
    if 'huggingface' in config:
        huggingface_config = config['huggingface']
        if 'loginToken' in huggingface_config:
            login(huggingface_config['loginToken'])
    if 'vllm' in config:
        if vllm_config:
            vllm_config = config['vllm'].update(vllm_config)
        else:
            vllm_config = config['vllm']

    llm = LLM(model=model_name, **vllm_config)
    return llm


class VLLMServer:
    def __init__(self, model_name: str, vllm_config:dict[str, any]=None, cuda:list=None):
        if vllm_config is None:
            vllm_config = {}

        yaml_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name + '.yaml')
        if not os.path.exists(yaml_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            # Create default config
            default_config = {
                'vllm': {
                    'tensor_parallel_size': 1,
                    'tool-call-parser': 'hermes'
                }
            }
            # Write the default config to the YAML file
            YAMLConfigManager.write_yaml(yaml_path, default_config)
            config = default_config
        else:
            config = YAMLConfigManager.read_yaml(yaml_path)

        if 'vllm' in config:
            config['vllm'].update(vllm_config)
            vllm_config = config['vllm']

        # Ensure we have a valid port
        if 'port' not in vllm_config:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', 0))
            free_port = s.getsockname()[1]
            s.close()
            vllm_config['port'] = free_port

        if 'model_name' in vllm_config:
            self.model_name = vllm_config['model_name']
        else:
            self.model_name = model_name
        self.port = vllm_config['port']

        # Build command
        cuda_env = ''
        if cuda:
            cuda_env = f'CUDA_VISIBLE_DEVICES={",".join(str(gpu) for gpu in cuda)}'
        cmd = ['python -m vllm.entrypoints.openai.api_server', '--model', model_name, '--enable-auto-tool-choice']
        for k, v in vllm_config.items():
            if isinstance(v, bool) and v:
                cmd.append(f'--{k}')
            else:
                cmd.append(f'--{k}')
                cmd.append(str(v))

        cmd_str = ' '.join(cmd)
        cmd_str = cuda_env + ' ' + cmd_str
        # Start process
        self.process = subprocess.Popen(cmd_str, shell=True)

        # Wait until the port can respond
        url = f'http://127.0.0.1:{self.port}/v1/models'
        for _ in range(3000):
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        self.client = OpenAI(
            base_url=f'http://localhost:{self.port}/v1',
            api_key='null',
        )

        # self.pid = get_pid_by_grep(cmd_str)+1
        self.pid = self.process.pid
        logging.info(f'{self.model_name} start at pid: {self.pid}')

    def chat(self, messages, **kwargs):
        '''
        Sends a chat request to the local vLLM server.
        Adjust parameters as you like.
        '''
        payload = kwargs
        payload['messages'] = messages
        payload['model'] = self.model_name
        resp = self.client.chat.completions.create(**payload)
        return resp

    def kill_server(self):
        '''
        Terminates the child process and its child processes.
        '''
        try:
            parent = psutil.Process(self.pid)
            # Recursively terminate all child processes
            for child in parent.children(recursive=True):
                child.kill()
            # Terminate the parent process
            parent.kill()
            logging.info(f'{self.model_name} terminated')
        except psutil.NoSuchProcess:
            logging.warning(f'Process with pid {self.pid} does not exist.')
