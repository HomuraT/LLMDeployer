import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from src.models.vllm_loader import VLLMServer
from src.utils.gpu_utils import find_available_gpu

# Assume these imports exist in your environment
# from src.models.vllm_loader import VLLMServer
# from src.models.gpu_utils import find_available_gpu

model_pool: Dict[str, Dict[str, Any]] = {}
model_creation_locks: Dict[str, threading.Lock] = {}
model_creation_condition = threading.Condition()
IDLE_TIMEOUT = timedelta(minutes=10) # Example: 10 minutes idle timeout
# Optional: Timeout for how long a model can be stuck in "CREATING"
CREATION_STUCK_TIMEOUT = timedelta(minutes=15) # Example: 15 minutes
MODEL_TTL_MINUTES = 60 # Time-to-live for inactive models

def get_or_create_model(model_name: str)->VLLMServer | None:
    """
    Retrieves an existing VLLMServer instance from the pool or creates a new one.
    Handles concurrent requests for the same model using locks and conditions.
    Returns None if creation fails or no suitable GPU is found.

    Args:
        model_name (str): The identifier of the model to get or create.

    Returns:
        Optional[VLLMServer]: The VLLMServer instance or None if unavailable/failed.
    """
    global model_pool, model_creation_locks, model_creation_condition

    with model_creation_condition:
        # Check if the model exists and is ready
        if model_name in model_pool:
            entry = model_pool[model_name]
            if entry["server"] != "CREATING":
                entry["last_access"] = datetime.now() # Update access time
                logging.info(f"Returning existing server instance for {model_name}")
                # Ensure we return the actual server instance
                return entry["server"] # type: ignore
            else:
                # Model is being created by another thread, wait for it
                logging.info(f"Model {model_name} is currently being created by another thread. Waiting...")
                # Loop while the entry exists and is marked as CREATING
                while model_name in model_pool and model_pool[model_name]["server"] == "CREATING":
                    # Always wait if the condition is met
                    model_creation_condition.wait()
                    # Re-check condition after waking up before potentially looping again

                # After the loop (meaning server is no longer "CREATING" or entry is gone)
                # Check the final state
                if model_name in model_pool and model_pool[model_name]["server"] != "CREATING":
                     logging.info(f"Model {model_name} creation finished (either by self or another thread). Returning instance.")
                     entry = model_pool[model_name]
                     entry["last_access"] = datetime.now()
                     # Ensure we return the actual server instance
                     server = entry["server"]
                     if isinstance(server, VLLMServer):
                         return server
                     else:
                         # Should not happen if logic is correct, but handle defensively
                         logging.error(f"Model {model_name} entry found but server is not a VLLMServer instance: {type(server)}. Returning None.")
                         # Clean up the bad entry?
                         if model_name in model_pool: del model_pool[model_name]
                         model_creation_condition.notify_all() # Notify others about the cleanup
                         return None

                else:
                     # This means the entry was removed or still 'CREATING' (latter shouldn't happen if loop exited)
                     logging.error(f"Waited for model {model_name}, but it's no longer in the pool or creation failed.")
                     return None # Creation failed or model removed while waiting

        # --- Model doesn't exist, initiate creation ---
        logging.info(f"Model {model_name} not found in pool. Initiating creation.")

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
            if creation_success and server_instance is not None:
                # Update pool with the actual server instance
                model_pool[model_name] = {"server": server_instance, "last_access": datetime.now()}
                logging.info(f"Model {model_name} successfully added to the pool.")
            else:
                # Creation failed, remove the 'CREATING' entry
                if model_name in model_pool and model_pool[model_name]["server"] == "CREATING":
                    del model_pool[model_name]
                logging.error(f"Model {model_name} creation failed. Removed placeholder from pool.")
            # Notify all waiting threads that this creation attempt is finished (success or fail)
            model_creation_condition.notify_all()

    # Return the created instance (or None if failed)
    return server_instance

def idle_cleaner():
    """
    Thread function to remove models that haven't been accessed for more than IDLE_TIMEOUT.
    Only attempts to kill actual server instances, skipping placeholders or invalid entries.
    Handles potential race conditions during iteration and removal.
    """
    while True:
        time.sleep(60) # Check every 60 seconds
        with model_creation_condition: # Acquire lock for safe access and modification
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

def cleanup_inactive_models():
    """
    Periodically checks the model pool and removes inactive models based on MODEL_TTL_MINUTES.
    Runs in a background thread.
    """
    global model_pool, model_creation_condition
    logging.info("Starting background thread for inactive model cleanup.")
    while True:
        try: # Add error handling for the loop itself
            time.sleep(60 * 5) # Check every 5 minutes

            with model_creation_condition: # Use the condition's lock to safely access pool
                now = datetime.now()
                inactive_models = []
                for model_name, entry in model_pool.items():
                    # Check only active servers, not "CREATING" placeholders
                    if entry["server"] != "CREATING":
                        if now - entry["last_access"] > timedelta(minutes=MODEL_TTL_MINUTES):
                            inactive_models.append(model_name)

                if inactive_models:
                    logging.info(f"Found inactive models exceeding TTL ({MODEL_TTL_MINUTES} min): {inactive_models}")

                for model_name in inactive_models:
                    if model_name in model_pool: # Double-check existence before removal
                        entry = model_pool[model_name]
                        server_instance = entry["server"]
                        logging.info(f"Removing inactive model {model_name} due to TTL.")
                        # Ensure it's a valid server instance before killing
                        if hasattr(server_instance, 'kill_server') and callable(server_instance.kill_server):
                             try:
                                 server_instance.kill_server()
                             except Exception as e:
                                 logging.error(f"Error killing inactive server {model_name}: {e}", exc_info=True)
                        else:
                            logging.warning(f"Attempted to remove inactive model {model_name}, but found unexpected server entry: {server_instance}")

                        # Remove from pool after attempting kill
                        del model_pool[model_name]
                        logging.info(f"Model {model_name} removed from pool.")
                    else:
                        logging.warning(f"Tried to remove inactive model {model_name}, but it was already gone.")

        except Exception as e:
            logging.error(f"Error in cleanup_inactive_models loop: {e}", exc_info=True)
            # Avoid busy-looping on persistent errors
            time.sleep(60)

# --- Background thread for cleanup ---
# Start the cleanup thread when this module is loaded
cleanup_thread = threading.Thread(target=cleanup_inactive_models, daemon=True)
cleanup_thread.start()

def cleanup_vllm_servers():
    """
    Iterates through the model pool and shuts down all active VLLM servers.
    Intended to be called on application shutdown (e.g., via atexit or signal handler).
    """
    global model_pool, model_creation_condition
    logging.info("Initiating VLLM server cleanup on application shutdown...")

    # Use the condition's lock to safely access and modify the pool
    with model_creation_condition:
        # Iterate over a copy of keys to avoid modification issues during iteration
        model_names = list(model_pool.keys())
        logging.info(f"Found models in pool to cleanup: {model_names}")

        for model_name in model_names:
            # Check existence again within the loop as map might change (though less likely with lock)
            if model_name in model_pool:
                entry = model_pool.get(model_name)
                # Check if the entry exists and contains a valid server instance
                if entry and isinstance(entry, dict) and "server" in entry:
                    server_instance = entry["server"]
                    # Check if it's an actual VLLMServer instance and not the "CREATING" placeholder
                    if hasattr(server_instance, 'kill_server') and callable(server_instance.kill_server):
                        logging.info(f"Shutting down VLLM server for model: {model_name}")
                        try:
                            # kill_server() should handle its own logging for PID etc.
                            server_instance.kill_server()
                        except Exception as e:
                            logging.error(f"Error shutting down server for {model_name}: {e}", exc_info=True)
                    elif server_instance == "CREATING":
                        logging.warning(f"Server for model {model_name} was still in CREATING state during global cleanup.")
                        # What to do here? The creating thread might still be running.
                        # Killing it might leave things inconsistent. Maybe just log it.
                    else:
                        logging.warning(f"Found unexpected server entry in model_pool for {model_name} during cleanup: {server_instance}")

                    # Remove from pool after attempting cleanup
                    # It's important to clear the pool so subsequent checks know it's gone
                    del model_pool[model_name]

                elif entry:
                    logging.warning(f"Found unexpected entry structure in model_pool for {model_name} during cleanup: {entry}")
                    # Also remove inconsistent entries
                    del model_pool[model_name]
            else:
                 logging.warning(f"Model {model_name} disappeared from pool during cleanup iteration.")

    logging.info("VLLM server global cleanup finished.")

# Example usage (commented out - should be started in app.py or similar)
# cleaner_thread = threading.Thread(target=idle_cleaner, daemon=True)
# cleaner_thread.start()
