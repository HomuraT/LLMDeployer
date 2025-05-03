import logging
import os
import signal
import subprocess
import socket
import time
import threading

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
    """
    Manages a VLLM OpenAI API server instance as a subprocess.
    Handles automatic startup, port finding, configuration loading,
    and graceful termination. Includes error handling for automatic restart.
    """
    MAX_RESTARTS = 3 # Maximum number of restart attempts

    def __init__(self, model_name: str, vllm_config: dict[str, any] = None, cuda: list = None):
        """
        Initializes and starts the VLLM server process.

        Args:
            model_name (str): The name of the model to load (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
            vllm_config (dict[str, any], optional): Additional VLLM configuration parameters. Defaults to None.
            cuda (list, optional): List of GPU IDs to use. Defaults to None.
        """
        self._vllm_config_override = vllm_config if vllm_config is not None else {}
        self._cuda_devices = cuda
        self._model_name_arg = model_name # Store the original model name argument for restarts
        self._restart_lock = threading.Lock()
        self._is_restarting = False
        self._restart_attempts = 0 # Initialize restart counter
        self.process = None
        self.pid = None
        self.client = None
        self.port = None # Will be set in _start_server_process
        self.model_name = None # Will be set in _start_server_process
        self.last_config = {} # Store the config used for the last successful start

        # Initial startup attempt
        try:
            self._start_server_process()
        except Exception as e:
            logging.error(f"Initial startup failed for {self._model_name_arg}: {e}")
            # Set state to indicate failure, but don't raise necessarily
            self.process = None
            self.pid = None
            self.port = None
            # Consider logging the server as unusable here or rely on caller checks

    def _load_config(self) -> dict:
        """
        Loads the YAML configuration for the model, applying overrides from constructor.

        Returns:
            dict: The final VLLM configuration dictionary.
        """
        yaml_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, self._model_name_arg + '.yaml')
        if not os.path.exists(yaml_path):
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            default_config = {'vllm': {'tensor_parallel_size': 1}} # Simplified default
            YAMLConfigManager.write_yaml(yaml_path, default_config)
            config = default_config
            logging.info(f"Created default config file at {yaml_path}")
        else:
            config = YAMLConfigManager.read_yaml(yaml_path)

        # Ensure 'vllm' key exists
        if 'vllm' not in config:
            config['vllm'] = {}

        # Apply overrides passed during instantiation
        config['vllm'].update(self._vllm_config_override)
        vllm_config = config['vllm']

        # Ensure we have a valid port
        return vllm_config

    def _start_server_process(self):
        """
        Internal method to start the VLLM server subprocess.
        Loads config, finds port, builds command, starts process, waits for ready.
        Raises:
            RuntimeError: If the server fails to start or become responsive.
        """
        with self._restart_lock: # Ensure only one thread starts/restarts at a time
            if self._is_restarting:
                 logging.info(f"Server {self._model_name_arg} is already restarting. Skipping.")
                 return # Avoid race condition if called concurrently

            # Load config each time to pick up potential manual changes
            vllm_config = self._load_config()
            self.last_config = vllm_config.copy() # Store for potential restarts

            # Ensure we have a valid port (find one if not specified or if previous failed)
            if 'port' not in vllm_config or self.port is None: # Also find new port if previous start failed
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', 0))
                        free_port = s.getsockname()[1]
                    vllm_config['port'] = free_port
                    logging.info(f"Assigned free port {free_port} for {self._model_name_arg}")
                except socket.error as e:
                    logging.error(f"Failed to find a free port: {e}")
                    raise RuntimeError("Could not bind to a free port") from e
            
            self.port = vllm_config['port']

            # Determine the actual model name to use (from config or constructor arg)
            self.model_name = vllm_config.get('model', self._model_name_arg) # Prefer config 'model' if present

            # Build command
            cuda_env = ''
            if self._cuda_devices:
                cuda_env = f'CUDA_VISIBLE_DEVICES={",".join(map(str, self._cuda_devices))}'
            # Base command - use self.model_name which might come from config
            cmd = ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', self.model_name]
            # Add other args from config, skipping 'model' as it's already included
            for k, v in vllm_config.items():
                if k == 'model': # Already handled
                    continue
                k_dashed = k.replace('_', '-') # Convert snake_case to kebab-case for CLI args
                if isinstance(v, bool):
                    if v:
                        cmd.append(f'--{k_dashed}')
                elif v is not None: # Append key and value if value is not None
                    cmd.append(f'--{k_dashed}')
                    cmd.append(str(v))

            cmd_str = ' '.join(cmd)
            full_cmd = f"{cuda_env} {cmd_str}" if cuda_env else cmd_str
            logging.info(f"Starting VLLM server for {self.model_name} with command: {full_cmd}")

            # Start process
            try:
                # Use Popen with shell=False for better control/security if possible
                # For simplicity with CUDA_VISIBLE_DEVICES, shell=True is kept for now
                self.process = subprocess.Popen(full_cmd, shell=True, start_new_session=True) # start_new_session for better cleanup
                self.pid = self.process.pid
                logging.info(f'{self.model_name} process starting with PID: {self.pid}')
            except Exception as e:
                logging.error(f"Failed to start subprocess for {self.model_name}: {e}")
                self.process = None
                self.pid = None
                raise RuntimeError(f"Subprocess Popen failed for {self.model_name}") from e

            # Wait until the port can respond
            url = f'http://127.0.0.1:{self.port}/health' # Use /health endpoint if available, fallback to /v1/models
            wait_start_time = time.time()
            max_wait_time = 300 # 5 minutes max wait time
            connected = False
            while time.time() - wait_start_time < max_wait_time:
                # Check if process terminated unexpectedly
                if self.process and self.process.poll() is not None:
                     logging.error(f"VLLM server process {self.pid} terminated unexpectedly during startup.")
                     self.kill_server() # Ensure cleanup
                     raise RuntimeError(f"VLLM server process {self.pid} terminated prematurely.")

                try:
                    r = requests.get(url, timeout=2)
                    # Also check /v1/models as a fallback
                    if r.status_code == 200:
                         models_url = f'http://127.0.0.1:{self.port}/v1/models'
                         try:
                             models_r = requests.get(models_url, timeout=2)
                             if models_r.status_code == 200:
                                 connected = True
                                 break
                         except requests.exceptions.RequestException:
                             pass # /v1/models might not be ready yet even if /health is
                    # If /health failed, try /v1/models directly
                    elif url == f'http://127.0.0.1:{self.port}/health':
                         models_url = f'http://127.0.0.1:{self.port}/v1/models'
                         try:
                             models_r = requests.get(models_url, timeout=2)
                             if models_r.status_code == 200:
                                 connected = True
                                 break
                         except requests.exceptions.RequestException:
                             pass # /v1/models failed too

                except requests.exceptions.ConnectionError:
                    pass # Server not up yet
                except requests.exceptions.Timeout:
                    logging.warning(f"Connection timeout while waiting for {self.model_name} at {url}. Retrying...")
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Request exception while waiting for {self.model_name}: {e}. Retrying...")

                time.sleep(2) # Wait longer between checks

            if not connected:
                logging.error(f"Failed to connect to VLLM server {self.model_name} at port {self.port} after {max_wait_time} seconds.")
                self.kill_server() # Ensure cleanup
                raise RuntimeError(f"VLLM server failed to start or become responsive on port {self.port}")

            # Server is up, create client
            self.client = OpenAI(
                base_url=f'http://localhost:{self.port}/v1',
                api_key='null', # vLLM OpenAI endpoint doesn't require a key
            )
            logging.info(f'{self.model_name} started successfully on port {self.port} with PID {self.pid}')
            self._restart_attempts = 0 # Reset restart counter on successful start


    def chat(self, messages: list[dict], **kwargs) -> object:
        """
        Sends a chat request to the running VLLM server.

        Args:
            messages (list[dict]): The list of messages for the chat completion.
            **kwargs: Additional arguments for the OpenAI completions API.

        Returns:
            object: The response object from openai.chat.completions.create.

        Raises:
            RuntimeError: If the server client is not initialized (server not running).
        """
        if not self.client:
            raise RuntimeError(f"VLLM server for {self.model_name} is not running or client not initialized.")
        # Ensure 'model' key is present, using the server's model name
        payload = kwargs
        payload['messages'] = messages
        payload['model'] = self.model_name # Use the model name associated with this server instance

        try:
            resp = self.client.chat.completions.create(**payload)
            return resp
        except Exception as e:
            logging.error(f"Error during chat completion request to {self.model_name}: {e}")
            # Consider if specific exceptions should trigger a restart check here too,
            # although the primary trigger is ConnectionError in the web app.
            raise # Re-raise the exception


    def handle_connection_error(self):
        """
        Handles a connection error by attempting to restart the server process.
        Uses a lock to prevent concurrent restarts and limits restart attempts.
        """
        if not self._restart_lock.acquire(blocking=False):
            logging.info(f"Restart already in progress for {self.model_name}. Skipping.")
            return

        try:
            if self._is_restarting:
                 logging.info(f"Restart flag already set for {self.model_name}. Skipping.")
                 return

            if self._restart_attempts >= VLLMServer.MAX_RESTARTS:
                logging.error(f"Max restart attempts ({VLLMServer.MAX_RESTARTS}) reached for {self.model_name}. Will not attempt further restarts.")
                # Optionally mark this server instance as permanently failed
                return # Do not restart

            self._is_restarting = True
            self._restart_attempts += 1 # Increment before attempting restart
            logging.warning(f"Connection error detected for {self.model_name}. Attempting restart {self._restart_attempts}/{VLLMServer.MAX_RESTARTS}...")

            try:
                self.kill_server() # Ensure the old process is gone
                time.sleep(2) # Brief pause before restarting
                logging.info(f"Restarting server process for {self.model_name}...")
                self._start_server_process() # Attempt to start again
                # Reset counter is now done inside _start_server_process on success
                # logging.info(f"Server {self.model_name} restarted successfully.") # Logged inside _start_server_process
            except Exception as e:
                logging.error(f"Failed to restart server {self.model_name} on attempt {self._restart_attempts}: {e}")
                # Restart failed, counter remains incremented. Next connection error will retry if limit not reached.

        finally:
            self._is_restarting = False # Allow future restarts if needed and limit not reached
            self._restart_lock.release()


    def kill_server(self):
        """
        Terminates the VLLM server subprocess and its children gracefully.
        Resets the state of the VLLMServer instance.
        """
        pid_to_kill = self.pid
        model_name_killed = self.model_name or self._model_name_arg # Use loaded name if available

        if pid_to_kill is None:
            logging.info(f"No active process PID found for {model_name_killed} to kill.")
            # Reset state even if no PID was found, in case process object exists
            self.process = None
            self.client = None
            self.port = None # Port might be reusable now
            # Keep self.last_config
            return

        logging.info(f"Attempting to terminate server {model_name_killed} (PID: {pid_to_kill})...")
        try:
            parent = psutil.Process(pid_to_kill)
            # Get children before killing parent
            children = parent.children(recursive=True)
            # Terminate children first
            for child in children:
                try:
                    child.terminate() # Try graceful termination first
                except psutil.NoSuchProcess:
                    pass # Child already gone
            # Wait a bit for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill() # Force kill remaining children
                    logging.warning(f"Force killed child process {p.pid} of {pid_to_kill}")
                except psutil.NoSuchProcess:
                    pass

            # Terminate the parent process
            try:
                parent.terminate() # Try graceful termination
                parent.wait(timeout=5) # Wait for termination
            except psutil.TimeoutExpired:
                logging.warning(f"Parent process {pid_to_kill} did not terminate gracefully. Killing...")
                parent.kill() # Force kill if necessary
            except psutil.NoSuchProcess:
                 logging.info(f"Parent process {pid_to_kill} already terminated.")


            logging.info(f'Server {model_name_killed} (PID: {pid_to_kill}) terminated.')

        except psutil.NoSuchProcess:
            logging.warning(f'Process with PID {pid_to_kill} not found during termination attempt for {model_name_killed}. It might have already exited.')
        except Exception as e:
            logging.error(f"Error during kill_server for {model_name_killed} (PID: {pid_to_kill}): {e}")
        finally:
            # Reset state regardless of termination success/failure
            self.process = None
            self.pid = None
            self.client = None
            self.port = None # Port is now potentially free
            logging.info(f"State reset for {model_name_killed} instance.")
