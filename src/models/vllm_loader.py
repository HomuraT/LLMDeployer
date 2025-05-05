import logging
import os
import signal
import subprocess
import socket
import time
import threading
# Add imports for prctl
import ctypes
import platform # To check OS

import psutil
import requests
from huggingface_hub import login
from openai import OpenAI
from tqdm import tqdm
try: # Make vllm import optional if needed elsewhere, though likely required here
    from vllm import LLM
except ImportError:
    LLM = None # type: ignore
    logging.warning("vLLM library not found. VLLMServer functionality will be limited.")

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
from src.utils.process_utils import get_pid_by_grep
from src.utils.yaml_utils import YAMLConfigManager
logging.basicConfig(level=logging.INFO)

# --- Additions for preexec_fn ---
IS_LINUX = platform.system() == "Linux"
PR_SET_PDEATHSIG = 1 # Linux specific constant
libc = None
prctl_syscall = None
can_use_prctl = False

if IS_LINUX:
    try:
        libc = ctypes.CDLL("libc.so.6")
        prctl_syscall = libc.prctl
        prctl_syscall.argtypes = [ctypes.c_int, ctypes.c_ulong]
        prctl_syscall.restype = ctypes.c_int
        can_use_prctl = True
        logging.info("Successfully loaded libc and prctl for PDEATHSIG.")
    except OSError as e:
        logging.warning(f"Could not load libc or find prctl: {e}. PDEATHSIG functionality disabled.")
        libc = None
        prctl_syscall = None
else:
    logging.info("Not running on Linux. PDEATHSIG functionality is not available.")

def set_pdeathsig_kill():
    """
    Sets the parent death signal to SIGKILL for the current process (Linux only).
    To be used as preexec_fn in subprocess.Popen.

    Input: None
    Output: None
    """
    # This function runs *in the child process* before exec.
    if IS_LINUX and can_use_prctl and prctl_syscall is not None:
        try:
            ret = prctl_syscall(PR_SET_PDEATHSIG, signal.SIGKILL)
            if ret != 0:
                # Try to get errno if possible, might be tricky in preexec_fn
                logging.warning(f"prctl(PR_SET_PDEATHSIG, SIGKILL) failed in child with return code {ret}")
        except Exception as e:
            # Logging might be difficult here depending on FD setup,
            # but attempt anyway. A failure here shouldn't prevent startup.
            logging.warning(f"Exception calling prctl in child: {e}")
# --- End Additions ---

def load_model(model_name:str, vllm_config=None)->LLM | None:
    # Add Type Hinting for return value
    """
    Loads a VLLM model based on configuration.

    :param model_name: Name of the model (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
    :type model_name: str
    :param vllm_config: Optional VLLM configuration overrides.
    :type vllm_config: dict | None
    :return: Loaded LLM object or None if vllm library is not available.
    :rtype: LLM | None
    """
    if LLM is None:
        logging.error("Cannot load model, vllm library not imported.")
        return None

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
            # Log specific error from _start_server_process
            logging.error(f"Initial startup failed for {self._model_name_arg}: {e}", exc_info=True) # Add traceback
            # Ensure consistent state after failure
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
        Uses preexec_fn for automatic child termination on Linux.

        :raises RuntimeError: If the server fails to start or become responsive.
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

            # --- Build command for shell=False ---
            env = os.environ.copy() # Start with current environment

            # --- Determine tensor_parallel_size from config (default 1) ---
            tp_size = vllm_config.get('tensor_parallel_size', 1)
            if 'tensor_parallel_size' not in vllm_config:
                 logging.info("tensor_parallel_size not found in config, defaulting to 1.")
            else:
                 logging.info(f"Using configured tensor_parallel_size={tp_size} from config.")

            # --- Select GPUs and set environment variables ---
            if self._cuda_devices: # Check if specific GPUs were pre-selected based on availability
                num_available_selected = len(self._cuda_devices)
                if num_available_selected < tp_size:
                    # Not enough available/selected GPUs to meet the requirement
                    error_msg = (f"Configuration requires tensor_parallel_size={tp_size}, "
                                 f"but only {num_available_selected} suitable GPUs ({self._cuda_devices}) were found/selected. "
                                 f"Cannot start VLLM server for {self.model_name}.")
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    # Select the first tp_size GPUs from the pre-selected list
                    gpus_to_use = self._cuda_devices[:tp_size]
                    env['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus_to_use))
                    env['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" # Ensure consistent ordering
                    logging.info(f"Selected {tp_size} GPU(s) {gpus_to_use} from available list {self._cuda_devices} to match tensor_parallel_size.")
                    logging.info(f"Setting CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} and CUDA_DEVICE_ORDER=PCI_BUS_ID for {self.model_name}")
            else:
                 # No specific GPUs were pre-selected (e.g., cuda=None passed to constructor)
                 # Let VLLM/CUDA handle GPU selection based on tp_size. Do not set env vars.
                 logging.info("No specific GPUs pre-selected. VLLM/CUDA will manage GPU allocation based on tensor_parallel_size.")

            # Base command - use self.model_name which might come from config
            # Ensure python executable is correctly found (e.g., use sys.executable)
            import sys
            cmd_list = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', self.model_name]

            # Add other args from config, skipping 'model' as it's already included
            for k, v in vllm_config.items():
                if k == 'model': # Already handled
                    continue
                k_dashed = k.replace('_', '-') # Convert snake_case to kebab-case for CLI args
                if isinstance(v, bool):
                    if v:
                        cmd_list.append(f'--{k_dashed}')
                elif v is not None: # Append key and value if value is not None
                    cmd_list.append(f'--{k_dashed}')
                    cmd_list.append(str(v))

            # --- Construct the executable command string for logging ---
            env_prefix = ""
            if 'CUDA_VISIBLE_DEVICES' in env:
                env_prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            # Safely quote arguments that might contain spaces or special characters
            import shlex
            cmd_str_for_log = env_prefix + " ".join(shlex.quote(arg) for arg in cmd_list)

            # --- Log the command ---
            logging.info(f"Starting VLLM server for {self.model_name}. Executable command:")
            logging.info(cmd_str_for_log) # Log the executable command separately for clarity
            # Original logging of just the list and env separately (can be kept or removed)
            # logging.info(f"Starting VLLM server for {self.model_name} with command list: {cmd_list}")
            # logging.info(f"Effective environment for subprocess includes: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

            # --- Start process with preexec_fn ---
            preexec_function = set_pdeathsig_kill if IS_LINUX and can_use_prctl else None
            if preexec_function:
                logging.info("Using preexec_fn to set PDEATHSIG on Linux.")
            else:
                logging.info("Not using preexec_fn (not Linux or prctl unavailable).")

            try:
                # Use shell=False, pass command as list, use env, and add preexec_fn
                # Keep start_new_session=True for independent process group if needed,
                # although PDEATHSIG provides termination guarantee.
                self.process = subprocess.Popen(
                    cmd_list,
                    env=env,
                    shell=False, # MUST be False for preexec_fn
                    start_new_session=True, # Creates independent process group
                    preexec_fn=preexec_function # Set parent death signal on Linux
                )
                self.pid = self.process.pid
                logging.info(f'{self.model_name} process starting with PID: {self.pid}')
            except Exception as e:
                logging.error(f"Failed to start subprocess for {self.model_name}: {e}", exc_info=True) # Add traceback
                self.process = None
                self.pid = None
                raise RuntimeError(f"Subprocess Popen failed for {self.model_name}") from e

            # Wait until the port can respond
            url = f'http://127.0.0.1:{self.port}/health' # Use /health endpoint if available, fallback to /v1/models
            wait_start_time = time.time()
            max_wait_time = 1200 # 5 minutes max wait time
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
                    # Check if process died while trying to connect
                    if self.process and self.process.poll() is not None:
                        logging.error(f"VLLM server process {self.pid} terminated unexpectedly while waiting for connection. Exit code: {self.process.returncode}")
                        self.kill_server() # Ensure cleanup
                        raise RuntimeError(f"VLLM server process {self.pid} terminated prematurely during startup.")
                    pass # Server not up yet, and process is still running (or None)
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
        Handles connection errors by attempting to restart the server.
        Includes locking and retry limits.
        """
        with self._restart_lock: # Acquire lock before checking/modifying state
            if self._is_restarting:
                logging.info(f"Restart already in progress for {self._model_name_arg}, ignoring concurrent error.")
                return # Another thread is handling the restart

            if self._restart_attempts >= self.MAX_RESTARTS:
                logging.error(f"Maximum restart attempts ({self.MAX_RESTARTS}) reached for {self._model_name_arg}. Server marked as failed.")
                # Optionally, mark this server instance as permanently failed in the pool
                self.kill_server() # Ensure it's dead
                # How to signal permanent failure? Maybe set pid/port to None and don't retry?
                # For now, just killing it prevents further restarts by this instance.
                # The pool logic might need adjustment if permanent failure is desired.
                return

            self._is_restarting = True # Mark that we are attempting a restart
            self._restart_attempts += 1
            logging.warning(f"Connection error detected for {self._model_name_arg}. Attempting restart ({self._restart_attempts}/{self.MAX_RESTARTS})...")

            # --- Perform Restart ---
            # 1. Ensure the old process is terminated
            logging.info(f"Killing existing server process for {self._model_name_arg} before restart...")
            self.kill_server() # Use the existing kill method
            time.sleep(2) # Give OS time to release resources (like port)

            # 2. Attempt to start a new server process using the last known good config
            logging.info(f"Attempting to restart {self._model_name_arg} using last known config...")
            try:
                # Reset relevant state variables before starting
                self.process = None
                self.pid = None
                # self.port might be reused if we found a free one, or kept if fixed in config.
                # _start_server_process will handle finding a new port if needed.
                self.client = None
                # We call _start_server_process which uses self.last_config internally now
                self._start_server_process() # This resets self._restart_attempts on success
                logging.info(f"Restart successful for {self._model_name_arg}.")
                # _start_server_process resets attempts, so no need here if successful

            except Exception as e:
                # Use exc_info=True for traceback
                logging.error(f"Restart attempt {self._restart_attempts} failed for {self._model_name_arg}: {e}", exc_info=True)
                # If startup fails again, kill_server was likely called within _start_server_process
                # or should be called to ensure cleanup
                self.kill_server() # Make sure it's cleaned up after failed restart attempt
            finally:
                # Only reset is_restarting flag. Attempts counter is handled by success/failure logic.
                self._is_restarting = False

        # Lock is released automatically by 'with' statement


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
                    # Reduce noise: logging.warning(f"Force killed child process {p.pid} of {pid_to_kill}")
                except psutil.NoSuchProcess:
                    pass

            # Terminate the parent process
            try:
                parent.terminate() # Try graceful termination
                parent.wait(timeout=5) # Wait for termination
                logging.info(f"Parent process {pid_to_kill} terminated gracefully.")
            except psutil.TimeoutExpired:
                logging.warning(f"Parent process {pid_to_kill} did not terminate gracefully. Killing...")
                parent.kill() # Force kill if necessary
                parent.wait() # Wait after kill
            except psutil.NoSuchProcess:
                 logging.info(f"Parent process {pid_to_kill} already terminated.")


            logging.info(f'Server {model_name_killed} (PID: {pid_to_kill}) termination process complete.')

        except psutil.NoSuchProcess:
            logging.warning(f'Process with PID {pid_to_kill} not found during termination attempt for {model_name_killed}. It might have already exited.')
        except Exception as e:
            logging.error(f"Error during kill_server for {model_name_killed} (PID: {pid_to_kill}): {e}", exc_info=True) # Add traceback
        finally:
            # Reset state regardless of termination success/failure
            self.process = None
            self.pid = None
            self.client = None
            self.port = None # Port might be reusable now
            # Keep self.last_config
            logging.info(f"State reset for {model_name_killed} instance.")
