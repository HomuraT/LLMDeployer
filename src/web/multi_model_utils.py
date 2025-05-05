import logging
import threading
import time
from datetime import datetime, timedelta

from src.models.vllm_loader import VLLMServer
from src.utils.gpu_utils import find_available_gpu

# Assume these imports exist in your environment
# from src.models.vllm_loader import VLLMServer
# from src.models.gpu_utils import find_available_gpu

model_pool = {}
model_pool_lock = threading.Lock()
# 为模型创建过程创建一个 Condition 变量
model_creation_condition = threading.Condition(model_pool_lock)
IDLE_TIMEOUT = timedelta(minutes=10) # Example: 10 minutes idle timeout
# Optional: Timeout for how long a model can be stuck in "CREATING"
CREATION_STUCK_TIMEOUT = timedelta(minutes=15) # Example: 15 minutes

def get_or_create_model(model_name: str)->VLLMServer | None:
    """
    Get a model if it exists, or create a new VLLMServer instance otherwise.
    Updates last access time on every call.
    Handles concurrent creation attempts using a Condition variable.
    Returns None if no suitable GPU is found or creation fails.
    """
    with model_creation_condition: # Acquire lock via Condition
        entry = model_pool.get(model_name)

        # Loop while the entry exists AND is marked as CREATING
        while entry and isinstance(entry, dict) and entry.get("server") == "CREATING":
            logging.info(f"Model {model_name} is being created by another thread, waiting...")
            # Wait for notification (releases lock, waits, re-acquires lock on wake-up)
            model_creation_condition.wait()
            # Re-fetch the entry after waking up to get the latest status
            entry = model_pool.get(model_name)
            logging.info(f"Woke up waiting for {model_name}. Current entry status: {entry.get('server') if isinstance(entry, dict) else entry}")

        # --- At this point, either the model exists and is ready, or it doesn't exist ---

        if entry and isinstance(entry.get("server"), VLLMServer):
            # Model exists and is a valid server instance
            logging.info(f"Found existing VLLMServer instance for {model_name}. Updating access time.")
            entry["last_access"] = datetime.now()
            return entry["server"]
        elif entry:
             # Entry exists but is not a valid server (e.g., unexpected state after wait)
             logging.error(f"Unexpected state for model {model_name} in pool after wait: {entry.get('server')}. Removing invalid entry.")
             # Remove the problematic entry
             if model_name in model_pool:
                 del model_pool[model_name]
             # Fall through to attempt creation again (or return None if creation isn't attempted below)
             # This path ideally shouldn't be hit if state management is correct.
             return None # Or decide if re-creation attempt is desired

        # --- Model does not exist in the pool or was invalid, proceed to create ---
        logging.info(f"Model {model_name} not found or invalid in pool. Initiating creation.")
        # Mark as CREATING under lock
        model_pool[model_name] = {"server": "CREATING", "last_access": datetime.now()}
        # The lock will be released after the 'with' block exits

    # --- Lock released, perform potentially long-running creation ---
    server_instance: VLLMServer | None = None
    creation_success = False
    try:
        available_gpus = find_available_gpu(model_name=model_name)
        if not available_gpus:
            logging.error(f"No suitable GPU found for model {model_name}. Cannot create server.")
            # server_instance remains None
        else:
            logging.info(f"Creating VLLMServer for {model_name} on GPU(s): {available_gpus}")
            # VLLMServer.__init__ contains the blocking _start_server_process call
            server_instance = VLLMServer(model_name, cuda=available_gpus)

            # Check if server initialization succeeded (VLLMServer.__init__ handles internal errors)
            # A successful init means the server is up and responding.
            if server_instance and server_instance.pid and server_instance.port and server_instance.client:
                 logging.info(f"VLLMServer for {model_name} created successfully (PID: {server_instance.pid}, Port: {server_instance.port}).")
                 creation_success = True
            else:
                 logging.error(f"VLLMServer initialization failed for {model_name} (PID/Port/Client check failed) despite available GPU.")
                 # Attempt cleanup if the instance exists but is invalid
                 if server_instance and hasattr(server_instance, 'kill_server'):
                     try:
                         server_instance.kill_server()
                     except Exception as kill_e:
                         logging.error(f"Error during cleanup of failed server instance for {model_name}: {kill_e}")
                 server_instance = None # Ensure we return None later

    except Exception as e:
        # Catch errors during find_available_gpu or VLLMServer() instantiation/startup
        logging.error(f"Exception during VLLMServer creation process for {model_name}: {e}", exc_info=True)
        # Attempt cleanup if the instance exists but is invalid
        if 'server_instance' in locals() and server_instance and hasattr(server_instance, 'kill_server'):
             try:
                 server_instance.kill_server()
             except Exception as kill_e:
                 logging.error(f"Error cleaning up failed server during exception for {model_name}: {kill_e}")
        server_instance = None # Ensure failure is recorded
        creation_success = False

    finally:
        # --- Re-acquire lock to update the pool and notify waiting threads ---
        with model_creation_condition:
            if creation_success and server_instance:
                # Creation succeeded, update the pool with the actual server instance
                logging.info(f"Updating model pool for successfully created {model_name}.")
                model_pool[model_name] = {
                    "server": server_instance,
                    "last_access": datetime.now() # Update access time upon creation
                }
            else:
                # Creation failed, remove the "CREATING" placeholder
                logging.warning(f"Failed to create VLLMServer for {model_name}. Removing placeholder from pool.")
                current_entry = model_pool.get(model_name)
                # Only remove if it's still the placeholder we added
                if current_entry and isinstance(current_entry, dict) and current_entry.get("server") == "CREATING":
                    del model_pool[model_name]
                # If it's something else, maybe another thread succeeded? Log it.
                elif current_entry:
                     logging.warning(f"Tried to remove placeholder for {model_name}, but found different state: {current_entry.get('server')}")

            # Notify all threads waiting on this condition (whether success or failure)
            logging.info(f"Notifying waiting threads for model {model_name} creation status.")
            model_creation_condition.notify_all()

            # Return the created instance or None if failed
            return server_instance

def idle_cleaner():
    """
    Thread function to remove models that haven't been accessed for more than IDLE_TIMEOUT.
    Only attempts to kill actual server instances, skipping placeholders or invalid entries.
    Handles potential race conditions during iteration and removal.
    """
    while True:
        time.sleep(60) # Check every 60 seconds
        with model_pool_lock: # Acquire lock for safe access and modification
            now = datetime.now()
            to_remove = [] # List of model names to remove

            # Iterate over a snapshot of model names to avoid issues with dict size changes during iteration
            current_model_names = list(model_pool.keys())

            for m_name in current_model_names:
                info = model_pool.get(m_name) # Get current info under lock

                # --- Basic validation of the entry ---
                if not isinstance(info, dict) or "server" not in info or "last_access" not in info:
                    logging.warning(f"Found malformed entry for '{m_name}' in model_pool during cleanup: {info}. Skipping.")
                    continue # Skip potentially corrupted entries

                server_instance = info.get("server")
                last_access_time = info.get("last_access")

                # --- Check 1: Idle timeout for valid server instances ---
                if isinstance(server_instance, VLLMServer) and isinstance(last_access_time, datetime):
                    if (now - last_access_time) > IDLE_TIMEOUT:
                        logging.info(f"Model {m_name} idle timeout ({IDLE_TIMEOUT}) reached (Last access: {last_access_time}). Scheduling for removal.")
                        to_remove.append(m_name)
                # --- Check 2: Stuck "CREATING" state ---
                elif server_instance == "CREATING" and isinstance(last_access_time, datetime):
                     if (now - last_access_time) > CREATION_STUCK_TIMEOUT:
                          logging.warning(f"Model {m_name} appears stuck in 'CREATING' state for over {CREATION_STUCK_TIMEOUT}. Scheduling for removal.")
                          to_remove.append(m_name) # Also remove potentially stuck creations

            # --- Perform removals after iteration ---
            for m_name in to_remove:
                info_to_remove = model_pool.get(m_name) # Re-get info under lock before acting

                if not info_to_remove:
                     logging.warning(f"Tried to remove '{m_name}', but it was already gone from the pool.")
                     continue

                server_instance_to_kill = info_to_remove.get("server")

                # Kill the server only if it's a valid VLLMServer instance
                if isinstance(server_instance_to_kill, VLLMServer):
                    logging.info(f"Attempting to kill idle/stuck server for model {m_name}.")
                    try:
                        server_instance_to_kill.kill_server()
                        logging.info(f"Successfully killed server for model {m_name}.")
                    except Exception as e:
                        logging.error(f"Error killing server for idle/stuck model {m_name}: {e}", exc_info=True)
                        # Continue to remove from pool even if killing fails

                # Always remove the entry from the pool after processing
                try:
                    if m_name in model_pool: # Check again before deleting
                        del model_pool[m_name]
                        logging.info(f"Removed entry for model {m_name} from pool.")
                except KeyError:
                     logging.warning(f"Tried to delete entry for '{m_name}', but it was already gone (potential race condition?).")

# Example usage (commented out - should be started in app.py or similar)
# cleaner_thread = threading.Thread(target=idle_cleaner, daemon=True)
# cleaner_thread.start()
