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
IDLE_TIMEOUT = timedelta(minutes=10)

def get_or_create_model(model_name: str)->VLLMServer | None:
    """
    Get a model if it exists, or create a new VLLMServer instance otherwise.
    Updates last access time on every call.
    Handles concurrent creation attempts using a Condition variable.
    Returns None if no suitable GPU is found or creation fails.
    """
    with model_creation_condition: # 通过 Condition 获取锁
        entry = model_pool.get(model_name)
        while entry and entry.get("server") == "CREATING":
            # 检查是否真的是 CREATING 状态 (避免键错误)
            # 模型正在被其他线程创建，等待通知
            logging.info(f"Model {model_name} is being created by another thread, waiting...")
            model_creation_condition.wait() # 释放锁，等待，被唤醒后重新获取锁
            # 唤醒后重新检查 entry
            entry = model_pool.get(model_name)

        if entry:
            # 模型存在 (且不再是 "CREATING" 状态)
            if isinstance(entry.get("server"), VLLMServer):
                entry["last_access"] = datetime.now()
                return entry["server"]
            else: # 处理意外状态
                logging.error(f"Unexpected state for model {model_name} in pool after wait: {entry.get('server')}")
                # 为安全起见，移除无效条目
                if model_name in model_pool:
                    del model_pool[model_name]
                return None # 表明失败
        else:
            # 模型未找到，标记为 CREATING，然后在锁外进行创建
            logging.info(f"Model {model_name} not found, initiating creation.")
            model_pool[model_name] = {"server": "CREATING", "last_access": datetime.now()}
            # 'with' 代码块结束后锁会自动释放

    # --- 锁已释放，执行创建 ---
    server_instance: VLLMServer | None = None
    creation_success = False
    try:
        available_gpus = find_available_gpu(model_name=model_name) # Pass model_name for potential memory adjustment
        if not available_gpus:
            logging.error(f"No suitable GPU found for model {model_name}. Cannot create server.")
            # 没有创建 server, server_instance 保持 None
        else:
            logging.info(f"Creating VLLMServer for {model_name} on GPU(s): {available_gpus}")
            try:
                server_instance = VLLMServer(model_name, cuda=available_gpus)
                # Check if server initialization failed internally
                if not server_instance or not server_instance.pid or not server_instance.port:
                     logging.error(f"VLLMServer initialization failed for {model_name} despite available GPU.")
                     if server_instance:
                         server_instance.kill_server() # Ensure partial resources are released
                     server_instance = None # Mark as failed
                else:
                    creation_success = True # 仅当实例有效时标记成功

            except Exception as e:
                 logging.error(f"Exception during VLLMServer creation for {model_name}: {e}", exc_info=True) # 添加 traceback
                 # If server object exists but is invalid, try to clean up
                 if 'server_instance' in locals() and server_instance and hasattr(server_instance, 'kill_server'):
                     try:
                         server_instance.kill_server()
                     except Exception as kill_e:
                         logging.error(f"Error cleaning up failed server during exception for {model_name}: {kill_e}")
                 server_instance = None # Mark as failed

    except Exception as outer_e:
        # 捕获 find_available_gpu 可能的异常
        logging.error(f"Error during GPU finding or server creation process for {model_name}: {outer_e}", exc_info=True)
        server_instance = None
        creation_success = False

    finally:
        # --- 重新获取锁以更新池并通知等待者 ---
        with model_creation_condition:
            if creation_success and server_instance:
                # 创建成功
                logging.info(f"Successfully created VLLMServer for {model_name}. Updating pool.")
                model_pool[model_name] = {
                    "server": server_instance,
                    "last_access": datetime.now()
                }
                model_creation_condition.notify_all() # 通知所有等待线程
                return server_instance
            else:
                # 创建失败
                logging.warning(f"Failed to create VLLMServer for {model_name}. Removing placeholder.")
                current_entry = model_pool.get(model_name)
                # 仅当条目仍然是 CREATING 占位符时才删除
                if current_entry and isinstance(current_entry, dict) and current_entry.get("server") == "CREATING":
                    del model_pool[model_name]
                model_creation_condition.notify_all() # 通知等待线程创建失败
                return None

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
