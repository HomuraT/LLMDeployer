import os
import signal
import subprocess
import socket
import time
import threading
import datetime
import sys
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
    # Import logger after log_config to avoid circular imports
    from src.utils.log_config import logger
    logger.warning("vLLM library not found. VLLMServer functionality will be limited.")

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
from src.utils.process_utils import get_pid_by_grep
from src.utils.yaml_utils import YAMLConfigManager
# Import the new logging system
from src.utils.log_config import logger, get_model_logger, cleanup_model_logger

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
        logger.info("Successfully loaded libc and prctl for PDEATHSIG.")
    except OSError as e:
        logger.warning(f"Could not load libc or find prctl: {e}. PDEATHSIG functionality disabled.")
        libc = None
        prctl_syscall = None
else:
    logger.info("Not running on Linux. PDEATHSIG functionality is not available.")

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
                logger.warning(f"prctl(PR_SET_PDEATHSIG, SIGKILL) failed in child with return code {ret}")
        except Exception as e:
            # Logging might be difficult here depending on FD setup,
            # but attempt anyway. A failure here shouldn't prevent startup.
            logger.warning(f"Exception calling prctl in child: {e}")
# --- End Additions ---

def load_model(model_name:str, vllm_config=None)->LLM | None:
    # Add Type Hinting for return value
    """
    Loads a VLLM model based on configuration.

    :param model_name: Name of the model (e.g., 'meta-llama/Llama-3.1-8B-Instruct' or 'qwen/Qwen-7B' for ModelScope).
    :type model_name: str
    :param vllm_config: Optional VLLM configuration overrides.
    :type vllm_config: dict | None
    :return: Loaded LLM object or None if vllm library is not available.
    :rtype: LLM | None
    """
    if LLM is None:
        logger.error("Cannot load model, vllm library not imported.")
        return None

    # --- ModelScope Integration ---
    # Check if we should use ModelScope. This could be driven by a global config or another mechanism.
    # For now, we assume if a model_name looks like a ModelScope ID (e.g., contains '/'),
    # or if a specific flag is set in vllm_config, we use ModelScope.
    # A more robust way would be a dedicated flag or checking model_name format.
    # Based on user request, we will set it directly.
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    logger.info("VLLM_USE_MODELSCOPE environment variable set to True for LLM loading.")
    # The model_name will now be treated as a ModelScope model ID by vLLM.
    # Ensure 'revision' and 'trust_remote_code' can be passed via vllm_config.
    # --- End ModelScope Integration ---

    if vllm_config is None:
        vllm_config = {}
    config_file_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name.replace('/', '_') + '.yaml') # Handle slashes in model_name for filename
    if not os.path.exists(config_file_path) and '/' in model_name: # Try original name if replacement not found
        config_file_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name + '.yaml')

    config = YAMLConfigManager.read_yaml(config_file_path) # Adjusted path

    if config and 'huggingface' in config: # Check if config is not None
        huggingface_config = config['huggingface']
        if 'loginToken' in huggingface_config:
            login(huggingface_config['loginToken'])
    
    final_vllm_params = {}
    if config and 'vllm' in config: # Check if config is not None
        final_vllm_params.update(config['vllm'])
    
    # Overwrite with any explicitly passed vllm_config
    final_vllm_params.update(vllm_config)

    # Ensure 'trust_remote_code' is present if not already, for ModelScope often needed.
    # This can be controlled via the YAML. Example adds it if not set.
    # if 'trust_remote_code' not in final_vllm_params:
    #     final_vllm_params['trust_remote_code'] = True
    #     logging.info("Setting 'trust_remote_code=True' by default for ModelScope compatibility.")


    # The model_name argument to LLM() will be the ModelScope ID.
    # Other parameters like 'revision', 'trust_remote_code' should be in final_vllm_params.
    llm = LLM(model=model_name, **final_vllm_params)
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
        
        # Create model-specific logger
        self.logger = get_model_logger(model_name)

        # Initial startup attempt
        try:
            self._start_server_process()
        except Exception as e:
            # Log specific error from _start_server_process
            self.logger.error(f"Initial startup failed for {self._model_name_arg}: {e}")
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
            self.logger.info(f"Created default config file at {yaml_path}")
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
                 self.logger.info(f"Server {self._model_name_arg} is already restarting. Skipping.")
                 return # Avoid race condition if called concurrently

            # Load config each time to pick up potential manual changes
            vllm_config = self._load_config()
            self.last_config = vllm_config.copy() # Store for potential restarts

            # Ensure we have a valid port (find one if not specified or if previous failed)
            if 'port' not in vllm_config or self.port is None: # Also find new port if previous start failed
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', 0))
                        free_port = s.getsockname()[1]
                    vllm_config['port'] = free_port
                    self.logger.info(f"Assigned free port {free_port} for {self._model_name_arg}")
                except socket.error as e:
                    self.logger.error(f"Failed to find a free port: {e}")
                    raise RuntimeError("Could not bind to a free port") from e
            
            self.port = vllm_config['port']

            # Determine the actual model name to use (from config or constructor arg)
            # This model_name will be passed to --model and should be a ModelScope ID
            self.model_name = vllm_config.get('model', self._model_name_arg) 

            # --- Build command for shell=False ---
            env = os.environ.copy() # Start with current environment

            # --- ModelScope Integration for VLLM Server ---
            env['VLLM_USE_MODELSCOPE'] = 'True'
            self.logger.info(f"VLLM_USE_MODELSCOPE environment variable set for VLLM server process for model {self.model_name}.")
            # Ensure 'trust_remote_code' and 'revision' are handled.
            # 'trust_remote_code' will be added as a command line arg if present in vllm_config.
            # 'revision' will also be added as a command line arg if present in vllm_config.
            # --- End ModelScope Integration ---


            # --- Determine tensor_parallel_size from config (default 1) ---
            tp_size = vllm_config.get('tensor_parallel_size', 1)
            if 'tensor_parallel_size' not in vllm_config:
                 self.logger.info("tensor_parallel_size not found in config, defaulting to 1.")
            else:
                 self.logger.info(f"Using configured tensor_parallel_size={tp_size} from config.")

            # --- Select GPUs and set environment variables ---
            if self._cuda_devices: # Check if specific GPUs were pre-selected based on availability
                num_available_selected = len(self._cuda_devices)
                if num_available_selected < tp_size:
                    # Not enough available/selected GPUs to meet the requirement
                    error_msg = (f"Configuration requires tensor_parallel_size={tp_size}, "
                                 f"but only {num_available_selected} suitable GPUs ({self._cuda_devices}) were found/selected. "
                                 f"Cannot start VLLM server for {self.model_name}.")
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    # Select the first tp_size GPUs from the pre-selected list
                    gpus_to_use = self._cuda_devices[:tp_size]
                    # Only set CUDA_VISIBLE_DEVICES if _cuda_devices is not empty,
                    # otherwise let VLLM handle it (e.g. for CPU execution or full auto selection)
                    if gpus_to_use:
                        env['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus_to_use))
                        env['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" # Ensure consistent ordering
                        self.logger.info(f"Selected {tp_size} GPU(s) {gpus_to_use} from available list {self._cuda_devices} to match tensor_parallel_size.")
                    else:
                        self.logger.info("No specific GPUs pre-selected or list was empty. VLLM/CUDA will manage GPU allocation.")

            else: # _cuda_devices is None or empty
                # Let VLLM/CUDA handle GPU selection based on tp_size. Do not set env vars.
                self.logger.info("No specific GPUs pre-selected. VLLM/CUDA will manage GPU allocation based on tensor_parallel_size.")

            cmd_list = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', self.model_name]

            # Add other parameters from vllm_config to the command list
            # This will now also include 'revision' and 'trust_remote_code' if they are in vllm_config
            for k, v in vllm_config.items():
                if k == 'model': # Already handled
                    continue
                if k == 'port': # Already handled
                    cmd_list.extend([f'--{k.replace("_", "-")}', str(v)]) # Use the port from vllm_config
                    continue

                param_name = f'--{k.replace("_", "-")}'
                if isinstance(v, bool):
                    if v: # Add flags like --trust-remote-code
                        cmd_list.append(param_name)
                elif isinstance(v, list): # Handle list arguments, e.g., --gpu-memory-utilization 0.9 0.8
                    cmd_list.append(param_name)
                    cmd_list.extend([str(item) for item in v])
                else: # Handle key-value arguments
                    cmd_list.extend([param_name, str(v)])
            
            # Ensure port is correctly passed if it was in vllm_config initially or dynamically assigned
            if '--port' not in cmd_list:
                 cmd_list.extend(['--port', str(self.port)])


            # self.logger.info(f"Starting VLLM server for {self.model_name}. Executable command (first few elements): {' '.join(cmd_list[:7])}...")
            # Full command can be very long, so log only a part or specific critical params.
            self.logger.info(f"Full VLLM server command: {' '.join(cmd_list)}")
            # self.logger.info(f"Environment for subprocess will include: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}, VLLM_USE_MODELSCOPE={env.get('VLLM_USE_MODELSCOPE')}")
            self.logger.info(f"VLLM server environment overrides: VLLM_USE_MODELSCOPE={env.get('VLLM_USE_MODELSCOPE')}, CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'Not Set/Let VLLM handle')}")


            # Setup log files for the subprocess
            # Sanitize model name for directory/file usage
            sanitized_model_name = self.model_name.replace('/', '_').replace('\\', '_')
            # Create base log directory if it doesn't exist
            base_log_dir = os.path.join(os.getcwd(), "vllm_logs")
            # Create model-specific log directory
            model_log_dir = os.path.join(base_log_dir, sanitized_model_name)
            os.makedirs(model_log_dir, exist_ok=True)

            # Get current timestamp for log file name
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Log file path will be determined after PID is known
            log_file = None
            log_handle = None

            try:
                # --- Start process with Popen ---
                # Temporarily disable redirection until we have PID if needed, or redirect immediately
                # For simplicity, let's redirect immediately. Log file name won't include PID yet.
                # This might be slightly less ideal if multiple processes start concurrently
                # A better way might involve getting PID first, then managing logs.
                # Let's try immediate redirection for now.

                # Define log file path (single file for both stdout and stderr)
                log_file = os.path.join(model_log_dir, f"{timestamp}.log")

                self.logger.info(f"Redirecting VLLM output to: {log_file}")

                # Open log file for writing
                log_handle = open(log_file, 'w')

                self.process = subprocess.Popen(
                    cmd_list,
                    env=env,
                    shell=False, # MUST be False for preexec_fn
                    start_new_session=True, # Creates independent process group
                    stdout=log_handle, # Redirect stdout
                    stderr=log_handle  # Redirect stderr to same file
                )
                self.pid = self.process.pid
                self.logger.info(f'{self.model_name} process starting with PID: {self.pid} and logs in {model_log_dir}') # Adjusted log message

                # Optional: Rename log file to include PID now that we have it
                # This might be less necessary now with timestamp uniqueness, but kept commented if needed
                # try:
                #     final_log = os.path.join(model_log_dir, f"{timestamp}_{self.pid}.log") # Example with PID
                #     # Need to close handle before renaming on some OS (e.g., Windows)
                #     if log_handle: log_handle.close()
                #     os.rename(log_file, final_log)
                #     # Reopen handle if needed, or adjust logic downstream
                #     log_file = final_log
                #     self.logger.info(f"Renamed VLLM log file to include PID: {self.pid}")
                # except OSError as rename_err:
                #     self.logger.warning(f"Could not rename log file to include PID {self.pid}: {rename_err}")
                #     # Reopen original handle if closed
                #     log_handle = open(log_file, 'a') # Reopen in append mode maybe?

            except Exception as e:
                self.logger.error(f"Failed to start subprocess for {self.model_name}: {e}")
                # --- Close handle if opened ---
                if log_handle:
                    log_handle.close()
                # --- End close handle ---
                self.process = None
                self.pid = None
                raise RuntimeError(f"Subprocess Popen failed for {self.model_name}") from e

            # Wait until the port can respond
            url = f'http://localhost:{self.port}/health' # Use /health endpoint if available, fallback to /v1/models
            wait_start_time = time.time()
            max_wait_time = 1200 # 5 minutes max wait time
            connected = False

            proxies = {
                "http": None,
                "https": None,
            }

            while time.time() - wait_start_time < max_wait_time:
                # Check if process terminated unexpectedly
                if self.process and self.process.poll() is not None:
                     self.logger.error(f"VLLM server process {self.pid} terminated unexpectedly during startup.")
                     self.kill_server() # Ensure cleanup
                     raise RuntimeError(f"VLLM server process {self.pid} terminated prematurely.")

                try:
                    r = requests.get(url, timeout=2, proxies=proxies)
                    # Also check /v1/models as a fallback
                    if r.status_code == 200:
                         models_url = f'http://localhost:{self.port}/v1/models'
                         try:
                             models_r = requests.get(models_url, timeout=2, proxies=proxies)
                             if models_r.status_code == 200:
                                 connected = True
                                 break
                         except requests.exceptions.RequestException:
                             pass # /v1/models might not be ready yet even if /health is
                    # If /health failed, try /v1/models directly
                    elif url == f'http://localhost:{self.port}/health':
                         models_url = f'http://localhost:{self.port}/v1/models'
                         try:
                             models_r = requests.get(models_url, timeout=2, proxies=proxies)
                             if models_r.status_code == 200:
                                 connected = True
                                 break
                         except requests.exceptions.RequestException:
                             pass # /v1/models failed too

                except requests.exceptions.ConnectionError:
                    # Check if process died while trying to connect
                    if self.process and self.process.poll() is not None:
                        self.logger.error(f"VLLM server process {self.pid} terminated unexpectedly while waiting for connection. Exit code: {self.process.returncode}")
                        self.kill_server() # Ensure cleanup
                        raise RuntimeError(f"VLLM server process {self.pid} terminated prematurely during startup.")
                    pass # Server not up yet, and process is still running (or None)
                except requests.exceptions.Timeout:
                    self.logger.warning(f"Connection timeout while waiting for {self.model_name} at {url}. Retrying...")
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Request exception while waiting for {self.model_name}: {e}. Retrying...")

                time.sleep(2) # Wait longer between checks

            if not connected:
                self.logger.error(f"Failed to connect to VLLM server {self.model_name} at port {self.port} after {max_wait_time} seconds.")
                self.kill_server() # Ensure cleanup
                raise RuntimeError(f"VLLM server failed to start or become responsive on port {self.port}")

            # Server is up, create client
            self.client = OpenAI(
                base_url=f'http://localhost:{self.port}/v1',
                api_key='null', # vLLM OpenAI endpoint doesn't require a key
            )
            self.logger.info(f'{self.model_name} started successfully on port {self.port} with PID {self.pid}')
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
            self.logger.error(f"Error during chat completion request to {self.model_name}: {e}")
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
                self.logger.info(f"Restart already in progress for {self._model_name_arg}, ignoring concurrent error.")
                return # Another thread is handling the restart

            if self._restart_attempts >= self.MAX_RESTARTS:
                self.logger.error(f"Maximum restart attempts ({self.MAX_RESTARTS}) reached for {self._model_name_arg}. Server marked as failed.")
                # Optionally, mark this server instance as permanently failed in the pool
                self.kill_server() # Ensure it's dead
                # How to signal permanent failure? Maybe set pid/port to None and don't retry?
                # For now, just killing it prevents further restarts by this instance.
                # The pool logic might need adjustment if permanent failure is desired.
                return

            self._is_restarting = True # Mark that we are attempting a restart
            self._restart_attempts += 1
            self.logger.warning(f"Connection error detected for {self._model_name_arg}. Attempting restart ({self._restart_attempts}/{self.MAX_RESTARTS})...")

            # --- Perform Restart ---
            # 1. Ensure the old process is terminated
            self.logger.info(f"Killing existing server process for {self._model_name_arg} before restart...")
            self.kill_server() # Use the existing kill method
            time.sleep(2) # Give OS time to release resources (like port)

            # 2. Attempt to start a new server process using the last known good config
            self.logger.info(f"Attempting to restart {self._model_name_arg} using last known config...")
            try:
                # Reset relevant state variables before starting
                self.process = None
                self.pid = None
                # self.port might be reused if we found a free one, or kept if fixed in config.
                # _start_server_process will handle finding a new port if needed.
                self.client = None
                # We call _start_server_process which uses self.last_config internally now
                self._start_server_process() # This resets self._restart_attempts on success
                self.logger.info(f"Restart successful for {self._model_name_arg}.")
                # _start_server_process resets attempts, so no need here if successful

            except Exception as e:
                # Use exc_info=True for traceback
                self.logger.error(f"Restart attempt {self._restart_attempts} failed for {self._model_name_arg}: {e}")
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
            self.logger.info(f"No active process PID found for {model_name_killed} to kill.")
            # Reset state even if no PID was found, in case process object exists
            self.process = None
            self.client = None
            self.port = None # Port might be reusable now
            # Keep self.last_config
            return

        self.logger.info(f"Attempting to terminate server {model_name_killed} (PID: {pid_to_kill})...")
        try:
            parent = psutil.Process(pid_to_kill)
            # Get children before killing parent
            children = parent.children(recursive=True)

            # --- Try SIGINT on parent first --- Allows VLLM potentially cleaner shutdown
            self.logger.info(f"Sending SIGINT to parent process {pid_to_kill}...")
            try:
                # Use os.kill for sending signals directly if psutil doesn't have .interrupt()
                # Or parent.send_signal(signal.SIGINT) if using newer psutil
                # Check psutil documentation for the best cross-platform way
                # For Linux/macOS, os.kill is reliable
                import os
                import signal
                os.kill(pid_to_kill, signal.SIGINT)
                parent.wait(timeout=5) # Wait up to 5 seconds for graceful exit
                self.logger.info(f"Parent process {pid_to_kill} potentially exited after SIGINT.")
            except psutil.TimeoutExpired:
                self.logger.warning(f"Parent process {pid_to_kill} did not exit after SIGINT within timeout.")
                # Proceed with child termination and more forceful parent termination
            except psutil.NoSuchProcess:
                self.logger.info(f"Parent process {pid_to_kill} exited before or during SIGINT wait.")
                parent = None # Mark parent as gone
            except Exception as sigint_err:
                self.logger.error(f"Error sending SIGINT to parent process {pid_to_kill}: {sigint_err}")

            # --- Terminate Children (if parent still exists or we need to be sure) ---
            # Re-fetch children if parent might have spawned new ones or some exited
            if parent and parent.is_running():
                 children = parent.children(recursive=True)
            else: # Parent is gone, find children based on original parent PID (less reliable)
                 # This part is tricky, maybe skip child termination if parent gone?
                 # For simplicity, let's assume children related to the *original* pid should be cleaned
                 # Re-finding children orphaned is hard, rely on original list
                 pass # Keep original children list

            if children:
                self.logger.info(f"Terminating child processes of {pid_to_kill}...")
                # Terminate children first
                for child in children:
                    try:
                        child.terminate() # Try graceful termination first (SIGTERM)
                    except psutil.NoSuchProcess:
                        pass # Child already gone
                # Wait a bit for children to terminate
                gone, alive = psutil.wait_procs(children, timeout=3)
                for p in alive:
                    try:
                        p.kill() # Force kill remaining children (SIGKILL)
                        # Reduce noise: self.logger.warning(f"Force killed child process {p.pid} of {pid_to_kill}")
                    except psutil.NoSuchProcess:
                        pass
                self.logger.info("Child process termination attempts complete.")

            # --- Terminate Parent (if still running) ---
            if parent and parent.is_running():
                 self.logger.info(f"Parent process {pid_to_kill} still running. Sending SIGTERM...")
                 try:
                     parent.terminate() # Try graceful termination (SIGTERM)
                     parent.wait(timeout=5) # Wait for termination
                     self.logger.info(f"Parent process {pid_to_kill} terminated gracefully after SIGTERM.")
                 except psutil.TimeoutExpired:
                     self.logger.warning(f"Parent process {pid_to_kill} did not terminate gracefully after SIGTERM. Sending SIGKILL...")
                     parent.kill() # Force kill if necessary (SIGKILL)
                     parent.wait() # Wait after kill
                     self.logger.info(f"Parent process {pid_to_kill} killed forcefully.")
                 except psutil.NoSuchProcess:
                     self.logger.info(f"Parent process {pid_to_kill} exited during SIGTERM wait.")
            elif parent is None:
                 # Parent already exited (likely after SIGINT or before we checked)
                 self.logger.info(f"Parent process {pid_to_kill} was already terminated.")

            self.logger.info(f'Server {model_name_killed} (PID: {pid_to_kill}) termination process complete.')

        except psutil.NoSuchProcess:
            self.logger.warning(f'Process with PID {pid_to_kill} not found during termination attempt for {model_name_killed}. It might have already exited.')
        except Exception as e:
            self.logger.error(f"Error during kill_server for {model_name_killed} (PID: {pid_to_kill}): {e}")
        finally:
            # --- Add cleanup for potential orphaned multiprocessing processes ---
            self.logger.info(f"Scanning for potential orphaned multiprocessing processes related to PID {pid_to_kill}...")
            cleaned_orphans = False
            try:
                # Iterate through all processes
                for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline')
                        if not cmdline: # Skip processes with no command line info
                            continue

                        # Check if it's a python multiprocessing orphan
                        is_python = 'python' in proc.info.get('name', '').lower()
                        is_orphan = proc.info.get('ppid') == 1
                        is_mp_process = any('multiprocessing.spawn' in arg or 'multiprocessing.resource_tracker' in arg for arg in cmdline)

                        # Heuristic: Check if it looks like a relevant orphan
                        # This is not perfect and might kill unrelated orphans if run concurrently
                        # Adding a check related to the model name/path in cmdline might help, but is fragile.
                        if is_python and is_orphan and is_mp_process:
                            self.logger.warning(f"Found potential orphaned multiprocessing process: PID={proc.info['pid']}, Cmd='{' '.join(cmdline)}'. Attempting termination.")
                            try:
                                proc.terminate() # Send SIGTERM
                                try:
                                    proc.wait(timeout=2)
                                except psutil.TimeoutExpired:
                                    self.logger.warning(f"Orphan process {proc.info['pid']} did not terminate after SIGTERM. Sending SIGKILL...")
                                    proc.kill() # Send SIGKILL
                                    proc.wait()
                                self.logger.info(f"Terminated potential orphan process {proc.info['pid']}.")
                                cleaned_orphans = True
                            except psutil.NoSuchProcess:
                                self.logger.info(f"Orphan process {proc.info['pid']} already exited.")
                            except Exception as orphan_kill_err:
                                self.logger.error(f"Error terminating orphan process {proc.info['pid']}: {orphan_kill_err}")

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue # Process disappeared or we lack permissions
                    except Exception as iter_err:
                         # Log errors during process iteration but continue if possible
                         self.logger.error(f"Error processing process PID {proc.info.get('pid', 'N/A')}: {iter_err}")

            except Exception as scan_err:
                self.logger.error(f"Error during orphan process scan: {scan_err}")

            if cleaned_orphans:
                 self.logger.info("Orphan process cleanup scan finished. GPU resources might take a moment to be fully released.")
            else:
                 self.logger.info("No suspected orphaned multiprocessing processes found.")
            # --- End orphan cleanup ---

            # Reset state regardless of termination success/failure
            self.process = None
            self.pid = None
            self.client = None
            self.port = None # Port might be reusable now
            # Keep self.last_config
            self.logger.info(f"State reset for {model_name_killed} instance.")
            
            # Clean up the model logger when server is killed
            cleanup_model_logger(self._model_name_arg)
