import re
# Remove subprocess import
# import subprocess
from src.utils.log_config import logger
try:
    import pynvml
except ImportError:
    pynvml = None
    logger.warning("pynvml library not found. GPU availability check will be limited.")

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil library not found. Username information for GPU processes will not be available.")

# Add docstring typing and PyCharm style comments
def find_available_gpu(model_name: str | None = None, min_memory_mb: int = 24000) -> list[int] | None:
    """
    Find available GPUs based on free memory using pynvml, optionally adjusting min_memory_mb.

    :param model_name: Optional; The name of the model (e.g., '7B') to potentially adjust memory requirements.
    :type model_name: str | None
    :param min_memory_mb: Minimum free memory required in Megabytes.
    :type min_memory_mb: int
    :return: A list of suitable GPU indices, or None if no suitable GPU is found or pynvml is unavailable.
    :rtype: list[int] | None
    """
    if not pynvml:
        logger.error("pynvml is not installed. Cannot perform GPU availability check.")
        return None # Cannot check GPUs without pynvml

    if model_name:
        # Regex to find a pattern like "7B", "13B", "30B", etc., from the model name
        pattern = r'(\d+(?:\.\d+)?)B'
        b_num = re.findall(pattern, model_name)
        if b_num:
            b_val = float(b_num[0])
            if b_val < 7:
                min_memory_mb = 10000  # smaller model => smaller memory requirement
            if b_val < 1:
                min_memory_mb = 3000   # very small model => even smaller
            logger.info(f"Adjusted minimum memory requirement to {min_memory_mb} MB for model {model_name}")


    selected_gpus = []
    max_memory = -1

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"Found {device_count} GPUs.")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory_mb = mem_info.free // (1024 * 1024) # Convert bytes to MB
                total_memory_mb = mem_info.total // (1024 * 1024) # Convert bytes to MB
                used_memory_mb = mem_info.used // (1024 * 1024) # Convert bytes to MB
                logger.info(f"GPU {i}: Used {used_memory_mb} MB / Total {total_memory_mb} MB (Free: {free_memory_mb} MB)")

                # Get running compute processes
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    if processes:
                        process_info = []
                        for p in processes:
                            try:
                                # Attempt to get process name (might require extra permissions or psutil)
                                # For now, just use PID and memory.
                                pid = p.pid
                                used_gpu_memory_mb = p.usedGpuMemory // (1024 * 1024) # Convert bytes to MB

                                username = f"PID {pid}" # Default to PID
                                if psutil:
                                    try:
                                        process = psutil.Process(pid)
                                        username = process.username()
                                    except psutil.NoSuchProcess:
                                        logger.warning(f"Process with PID {pid} not found by psutil on GPU {i}.")
                                        username = f"PID {pid} (gone?)" # Mark if process disappeared
                                    except psutil.AccessDenied:
                                        logger.warning(f"Access denied when trying to get username for PID {pid} on GPU {i}.")
                                        username = f"PID {pid} (no access)"
                                    except Exception as e:
                                        logger.warning(f"Error getting username for PID {pid} on GPU {i}: {e}")
                                        username = f"PID {pid} (error)"

                                process_info.append(f"{username} ({used_gpu_memory_mb} MB)")
                            except pynvml.NVMLError as proc_err:
                                logger.warning(f"Could not get details for a process on GPU {i}: {proc_err}")
                        if process_info:
                             logger.info(f"  Processes on GPU {i}: {'; '.join(process_info)}")
                    else:
                        logger.info(f"  No running compute processes found on GPU {i}.")
                except pynvml.NVMLError as e:
                    logger.warning(f"Could not get process info for GPU {i}: {e}")

                # Select the GPU(s) with the most memory that satisfy the minimum requirement
                if free_memory_mb >= min_memory_mb:
                    if free_memory_mb > max_memory:
                        max_memory = free_memory_mb
                        selected_gpus = [i] # Start a new list with this better GPU
                    elif free_memory_mb == max_memory:
                         selected_gpus.append(i) # Add GPU if it has the same max memory

            except pynvml.NVMLError as e:
                logger.warning(f"Could not get memory info for GPU {i}: {e}")
                continue # Skip this GPU if info is unavailable

        if selected_gpus:
            logger.info(f"GPU(s) {selected_gpus} selected with >= {min_memory_mb} MB free memory (Max found: {max_memory} MB).")
            return selected_gpus
        else:
            logger.info(f"No GPU found with minimum memory of {min_memory_mb} MB free.")
            return None

    except pynvml.NVMLError as e:
        logger.error(f"Failed to initialize NVML or get device count: {e}")
        return None
    finally:
        # Ensure NVML is shut down even if errors occur
        if pynvml:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                logger.error(f"Failed to shutdown NVML: {e}")

# Remove the old implementation using subprocess
# try:
#     result = subprocess.run(
#         ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#         timeout=10 # Add a timeout
#     )
#     if result.returncode != 0:
#         logging.error(f"Failed to execute nvidia-smi (Exit code {result.returncode}): {result.stderr}")
#         return None
#     if not result.stdout.strip():
#         logging.warning("nvidia-smi returned empty output for free memory.")
#         return None
#
#     free_memory = [int(x) for x in result.stdout.strip().split('\n')]
#     max_memory = -1
#     selected_gpus = []
#
#     for i, mem in enumerate(free_memory):
#         if mem >= min_memory_mb: # Check against minimum requirement first
#             if mem > max_memory:
#                 max_memory = mem
#                 selected_gpus = [i] # Found a new best GPU
#             elif mem == max_memory:
#                 selected_gpus.append(i) # Found another GPU with the same max memory
#
#     if selected_gpus:
#         logging.info(f"GPU(s) {selected_gpus} selected with >= {min_memory_mb} MB free memory (Max found: {max_memory} MB).")
#         return selected_gpus
#     else:
#         logging.info(f"No GPU found with minimum memory of {min_memory_mb} MB free.")
#         return None
#
# except FileNotFoundError:
#     logging.error("nvidia-smi command not found. Please ensure NVIDIA drivers and tools are installed.")
#     return None
# except subprocess.TimeoutExpired:
#     logging.error("nvidia-smi command timed out after 10 seconds. GPU might be unresponsive.")
#     return None
# except ValueError as e:
#     logging.error(f"Error parsing nvidia-smi output: {e}. Output was: '{result.stdout}'")
#     return None
# except Exception as e:
#     logging.error(f"Error finding available GPU via nvidia-smi: {e}")
#     return None