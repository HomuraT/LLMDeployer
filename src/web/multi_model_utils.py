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
IDLE_TIMEOUT = timedelta(minutes=10)

def get_or_create_model(model_name: str)->VLLMServer | None:
    """
    Get a model if it exists, or create a new VLLMServer instance otherwise.
    Updates last access time on every call.
    Returns None if no suitable GPU is found for creation.
    """
    with model_pool_lock:
        entry = model_pool.get(model_name)
        if not entry:
            available_gpus = find_available_gpu(model_name=model_name) # Pass model_name for potential memory adjustment
            if not available_gpus:
                logging.error(f"No suitable GPU found for model {model_name}. Cannot create server.")
                return None # Explicitly return None if no GPU found

            logging.info(f"Creating VLLMServer for {model_name} on GPU(s): {available_gpus}")
            try:
                server = VLLMServer(model_name, cuda=available_gpus)
                # Check if server initialization failed internally (e.g., process start failed)
                if not server.pid or not server.port:
                     logging.error(f"VLLMServer initialization failed for {model_name} despite available GPU.")
                     # Optional: Attempt cleanup if server object exists but is invalid
                     if server:
                         server.kill_server() # Ensure partial resources are released
                     return None

                model_pool[model_name] = {
                    "server": server,
                    "last_access": datetime.now()
                }
                entry = model_pool[model_name] # Update entry to the newly created one
            except Exception as e:
                 logging.error(f"Exception during VLLMServer creation for {model_name}: {e}")
                 return None # Return None if server creation throws an exception

        # Update last access time only if entry exists (either found or created)
        entry["last_access"] = datetime.now()
        return entry["server"]

def idle_cleaner():
    """
    Thread function to remove models that haven't been accessed for more than IDLE_TIMEOUT.
    """
    while True:
        time.sleep(60)  # check every 60 seconds
        with model_pool_lock:
            to_remove = []
            for m_name, info in model_pool.items():
                if datetime.now() - info["last_access"] > IDLE_TIMEOUT:
                    to_remove.append(m_name)
            for m_name in to_remove:
                model_pool[m_name]["server"].kill_server()
                del model_pool[m_name]

# Start idle_cleaner as a daemon thread
# cleaner_thread = threading.Thread(target=idle_cleaner, daemon=True)
# cleaner_thread.start()

# Example usage in request handler (pseudo-code):
# def handle_request(model_name, user_input):
#     llm_server = get_or_create_model(model_name)
#     # do something, e.g. llm_server.chat(...)
