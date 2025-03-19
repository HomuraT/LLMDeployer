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

def get_or_create_model(model_name: str)->VLLMServer:
    """
    Get a model if it exists, or create a new VLLMServer instance otherwise.
    Updates last access time on every call.
    """
    with model_pool_lock:
        entry = model_pool.get(model_name)
        if not entry:
            server = VLLMServer(model_name, cuda=find_available_gpu())
            model_pool[model_name] = {
                "server": server,
                "last_access": datetime.now()
            }
        else:
            entry["last_access"] = datetime.now()
        return model_pool[model_name]["server"]

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
