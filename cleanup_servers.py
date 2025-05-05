import logging
import os
import sys
import signal
import time
import errno
from typing import Optional, NoReturn

# Dynamically add the project root and parent to sys.path if necessary
# This helps ensure imports work correctly regardless of execution location.
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    # Also add the parent directory if src is directly under it (common structure)
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
except NameError:
    # __file__ might not be defined in some environments (e.g., interactive)
    logging.warning("Could not automatically determine project root. Imports might fail if PYTHONPATH is not set.")


# Import the necessary utility functions
try:
    # Attempt to set python path if the utility exists
    try:
        from src.utils.enviroment_utils import set_python_path
        set_python_path() # Ensure PYTHONPATH is set correctly for imports below
    except ImportError:
        logging.warning("`set_python_path` utility not found or failed. Proceeding without it.")

    from src.web.multi_model_utils import cleanup_vllm_servers
    from src.utils.process_utils import cleanup_potential_vllm_orphans
except ImportError as e:
    logging.error(f"Error importing necessary modules: {e}", exc_info=True)
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print("Please ensure the script is run from a location where 'src' package is discoverable, or that PYTHONPATH is configured correctly.", file=sys.stderr)
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)
logger = logging.getLogger(__name__) # Use a specific logger

# Define the expected PID file name
PID_FILENAME = "run_api.pid"
# Assume PID file is in the same directory as this script or project root
try:
    # Use PROJECT_ROOT determined earlier if available
    PID_FILEPATH = os.path.join(PROJECT_ROOT, PID_FILENAME)
except NameError:
    # Fallback if PROJECT_ROOT couldn't be determined
    logger.warning("PROJECT_ROOT not defined, assuming PID file is in the current directory.")
    PID_FILEPATH = PID_FILENAME


def get_pid_from_file(pid_filepath: str) -> Optional[int]:
    """
    Reads the PID from the specified file.

    从指定文件读取PID。

    :param pid_filepath: Path to the PID file.
    :type pid_filepath: str
    :return: The PID if found and valid, otherwise None.
    :rtype: Optional[int]
    """
    try:
        with open(pid_filepath, 'r') as f:
            pid_str = f.read().strip()
            if pid_str.isdigit():
                return int(pid_str)
            else:
                logger.warning(f"PID file '{pid_filepath}' contained non-numeric value: '{pid_str}'")
                return None
    except FileNotFoundError:
        logger.info(f"PID file '{pid_filepath}' not found. Assuming run_api is not running or didn't create the file.")
        return None
    except IOError as e:
        logger.error(f"Error reading PID file '{pid_filepath}': {e}", exc_info=True)
        return None
    except ValueError:
         logger.error(f"Error converting PID '{pid_str}' to integer in file '{pid_filepath}'.", exc_info=True)
         return None


def terminate_process(pid: int, signal_to_send: signal.Signals = signal.SIGTERM) -> bool:
    """
    Sends a signal to terminate the process with the given PID.

    向指定PID的进程发送终止信号。

    :param pid: The process ID to terminate.
    :type pid: int
    :param signal_to_send: The signal to send (default: SIGTERM).
    :type signal_to_send: signal.Signals
    :return: True if the signal was sent successfully, False otherwise.
    :rtype: bool
    """
    if pid <= 0:
        logger.warning(f"Invalid PID ({pid}) provided for termination.")
        return False
    try:
        os.kill(pid, signal_to_send)
        logger.info(f"Sent signal {signal_to_send.name} to process {pid}.")
        return True
    except ProcessLookupError:
        logger.info(f"Process {pid} not found. It might have already exited.")
        return True # Consider success if process doesn't exist
    except PermissionError:
        logger.error(f"Permission denied to send signal to process {pid}.")
        return False
    except OSError as e:
        # Handle specific OS errors like invalid argument (if pid is weird)
        if e.errno == errno.ESRCH: # No such process
             logger.info(f"Process {pid} not found (ESRCH). It might have already exited.")
             return True
        elif e.errno == errno.EPERM: # Operation not permitted
            logger.error(f"Permission denied to send signal to process {pid} (EPERM).")
            return False
        else:
            logger.error(f"Error sending signal {signal_to_send.name} to process {pid}: {e}", exc_info=True)
            return False

def remove_pid_file(pid_filepath: str) -> None:
    """
    Removes the PID file if it exists.

    如果PID文件存在，则删除它。

    :param pid_filepath: Path to the PID file.
    :type pid_filepath: str
    :return: None
    :rtype: None
    """
    try:
        if os.path.exists(pid_filepath):
            os.remove(pid_filepath)
            logger.info(f"Removed PID file '{pid_filepath}'.")
    except OSError as e:
        logger.error(f"Error removing PID file '{pid_filepath}': {e}", exc_info=True)


def main() -> None:
    """
    Main function to execute the cleanup process for the main API server,
    VLLM servers, and potential orphans.

    主函数，执行主API服务器、VLLM服务器和潜在孤儿进程的清理过程。

    :return: None
    :rtype: None
    """
    logger.info("Starting cleanup process...")

    # 1. Terminate the main run_api.py process
    logger.info(f"Attempting to terminate main API process using PID file: {PID_FILEPATH}")
    api_pid = get_pid_from_file(PID_FILEPATH)
    if api_pid:
        if terminate_process(api_pid, signal.SIGTERM):
            # Optional: Wait a moment for graceful shutdown
            time.sleep(2)
            # Check if it's still alive (optional, SIGTERM handler in run_api should handle cleanup)
            # If still alive, maybe send SIGKILL (use with caution)
            # terminate_process(api_pid, signal.SIGKILL)
            pass # Assuming SIGTERM is handled gracefully by run_api.py
        # Always try to remove the PID file if we found a PID
        remove_pid_file(PID_FILEPATH)
    else:
        logger.info("No valid PID found for main API process.")


    # 2. Clean up potential orphan VLLM processes (do this before shutting down managers)
    try:
        logger.info("Cleaning up potential orphan VLLM processes...")
        cleanup_potential_vllm_orphans()
        logger.info("Orphan cleanup finished successfully.")
    except Exception as orphan_cleanup_err:
        logger.error(f"Error during orphan process cleanup: {orphan_cleanup_err}", exc_info=True)

    # 3. Shut down active VLLM server managers and their processes
    try:
        logger.info("Shutting down active VLLM server managers and their processes...")
        cleanup_vllm_servers() # This function likely handles multiple servers
        logger.info("Active server cleanup finished successfully.")
    except Exception as active_cleanup_err:
        logger.error(f"Error during active server cleanup: {active_cleanup_err}", exc_info=True)

    logger.info("Cleanup process completed.")

if __name__ == "__main__":
    main() 