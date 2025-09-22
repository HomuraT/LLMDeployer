from src.utils.enviroment_utils import huggingface_use_domestic_endpoint, set_python_path # Keep if needed elsewhere
huggingface_use_domestic_endpoint()
set_python_path()

# import uvicorn # Not needed for Flask
# from fastapi import FastAPI # Not needed for Flask
import os
import sys
# Add imports for signal handling and cleanup
import signal
import atexit

# Dynamically add the project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assuming run_api.py is in the root
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the new logging system
from src.utils.log_config import logger, set_debug_mode

# Import the cleanup function
from src.web.multi_model_utils import cleanup_vllm_servers
# Import the run function or app object from the Flask app file
from src.web.app import run as run_flask_app # Import the run function from app.py
# from src.web.app import app as flask_app # Alternative if run() doesn't exist or you prefer direct control
from src.utils.process_utils import cleanup_potential_vllm_orphans # Import the new cleanup function

# --- PID File Configuration ---
PID_FILENAME = "run_api.pid"
PID_FILEPATH = os.path.join(PROJECT_ROOT, PID_FILENAME)

def remove_pid_file() -> None:
    """Removes the PID file if it exists. 安全地移除PID文件。"""
    try:
        if os.path.exists(PID_FILEPATH):
            os.remove(PID_FILEPATH)
            logger.info(f"Removed PID file {PID_FILEPATH} on exit.")
    except OSError as e:
        logger.error(f"Error removing PID file {PID_FILEPATH}: {e}")
# --- End PID File Configuration ---


# --- Signal Handling and Cleanup Registration --- (Modified)
def handle_signal(signum, frame):
    """Signal handler for SIGINT and SIGTERM."""
    logger.warning(f"Received signal {signal.Signals(signum).name}. Initiating VLLM server shutdown...")
    # It's crucial cleanup happens BEFORE Flask/Werkzeug completely exits if possible
    cleanup_vllm_servers() # Call the cleanup function from multi_model_utils
    remove_pid_file() # Remove PID file before exiting
    logger.info("Cleanup complete. Exiting run_api process.")
    # Allow the signal to potentially propagate or exit cleanly
    # Forceful exit might prevent Flask's own cleanup
    sys.exit(0)

# Register handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
try:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info("Registered signal handlers for SIGINT and SIGTERM.")
except ValueError as e:
    # This can happen if running in a thread where signals can't be registered
    logger.warning(f"Could not register signal handlers (may be running in a non-main thread): {e}")

# Register the PID file removal function to be called on normal exit or unhandled exceptions
# atexit cleanup_vllm_servers is removed as it's called within the signal handler now
atexit.register(remove_pid_file)
logger.info(f"Registered atexit cleanup function for PID file {PID_FILEPATH}.")
# --- End Cleanup Registration ---

# Remove FastAPI specific code
# app = FastAPI()
# app.include_router(chat_router)

if __name__ == "__main__":
    # --- Write PID file ---    
    current_pid = os.getpid()
    try:
        with open(PID_FILEPATH, 'w') as f:
            f.write(str(current_pid))
        logger.info(f"Main API process started with PID {current_pid}. PID written to {PID_FILEPATH}")
        # Note: atexit registration for remove_pid_file is done above, no need to repeat here
    except IOError as e:
        logger.error(f"Failed to write PID file {PID_FILEPATH}: {e}")
        logger.warning("Proceeding without a PID file. Cleanup script might not be able to terminate this process automatically.")
        # Decide if you want to exit if PID file cannot be written
        # sys.exit(1)
    # --- End Write PID file ---

    # --- Perform orphan cleanup at startup ---
    try:
        cleanup_potential_vllm_orphans()
    except Exception as initial_cleanup_err:
        logger.error(f"Error during initial orphan cleanup: {initial_cleanup_err}")
    # --- End initial cleanup ---

    # Call the run function from src/web/app.py
    logger.info(f"Starting Flask server via src.web.app.run()...")
    try:
        run_flask_app()
    except Exception as flask_err:
        logger.critical(f"Flask server encountered a fatal error: {flask_err}")
        # Ensure cleanup runs even if flask crashes badly, then exit
        # Signal handler might not have been called
        cleanup_vllm_servers()
        remove_pid_file()
        sys.exit(1) # Exit with error status
    finally:
        # This block might be reached if run_flask_app returns normally
        # or after the try/except block handles an error
        logger.info("Flask server process is ending.")
        # Ensure cleanup happens on normal return too (atexit should cover this, but belt-and-suspenders)
        # cleanup_vllm_servers() # Potentially redundant if signal/atexit works
        # remove_pid_file() # Already registered with atexit

    # Code here might not be reached if Flask runs indefinitely and exits via signal/error
    logger.info("Flask server has stopped gracefully.") # Might only log if Flask run returns
