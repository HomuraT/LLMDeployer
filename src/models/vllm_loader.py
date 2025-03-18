import os
import subprocess
import socket
import time

import requests
from huggingface_hub import login
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
from src.utils.yaml_utils import YAMLConfigManager

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

        config = YAMLConfigManager.read_yaml(os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name + '.yaml'))
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
            cuda_env = f'CUDA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in cuda)}'
        cmd = ['vllm', 'serve', model_name, '--enable-auto-tool-choice']
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
        self.pid = self.process.pid

        # Wait until the port can respond
        url = f'http://127.0.0.1:{self.port}/v1/models'
        for _ in tqdm(range(3000), desc='wait for model loading'):
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
        Terminates the child process.
        '''
        self.process.terminate()
