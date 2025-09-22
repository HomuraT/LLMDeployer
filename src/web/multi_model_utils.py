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
# 全局锁，用于保护整个get_or_create_model方法
model_global_lock = threading.Lock()
IDLE_TIMEOUT = timedelta(minutes=10) # Example: 10 minutes idle timeout
MODEL_TTL_MINUTES = 60 # Time-to-live for inactive models

def get_or_create_model(model_name: str) -> VLLMServer | None:
    """
    获取或创建VLLMServer实例。
    使用全局锁保护整个方法执行过程，确保线程安全。
    
    Args:
        model_name (str): 要获取或创建的模型名称标识符

    Returns:
        Optional[VLLMServer]: VLLMServer实例，如果获取/创建失败则返回None
    """
    global model_pool, model_global_lock

    # 整个方法使用全局锁保护
    with model_global_lock:
        logging.info(f"获取模型锁成功，开始处理模型: {model_name}")
        
        # 检查模型是否已存在
        if model_name in model_pool:
            entry = model_pool[model_name]
            server_instance = entry.get("server")
            
            # 验证是否为有效的VLLMServer实例
            if isinstance(server_instance, VLLMServer):
                entry["last_access"] = datetime.now()  # 更新访问时间
                logging.info(f"返回现有的服务器实例: {model_name}")
                return server_instance
            else:
                # 清理无效条目
                logging.warning(f"发现无效的模型条目: {model_name}，类型: {type(server_instance)}，正在清理")
                del model_pool[model_name]

        # 模型不存在，开始创建新实例
        logging.info(f"模型 {model_name} 不存在于池中，开始创建新实例")
        
        server_instance: VLLMServer | None = None
        try:
            # 查找可用GPU
            available_gpus = find_available_gpu(model_name=model_name)
            if not available_gpus:
                logging.error(f"未找到适合模型 {model_name} 的GPU，无法创建服务器")
                return None
            
            logging.info(f"为模型 {model_name} 在GPU {available_gpus} 上创建VLLMServer")
            
            # 创建VLLMServer实例（这是阻塞操作）
            server_instance = VLLMServer(model_name, cuda=available_gpus)
            
            # 验证服务器初始化是否成功
            if (server_instance and 
                server_instance.pid and 
                server_instance.port and 
                server_instance.client):
                
                # 成功创建，添加到模型池
                model_pool[model_name] = {
                    "server": server_instance, 
                    "last_access": datetime.now()
                }
                logging.info(f"VLLMServer创建成功: {model_name} "
                           f"(PID: {server_instance.pid}, Port: {server_instance.port})")
                return server_instance
            else:
                logging.error(f"VLLMServer初始化失败: {model_name} "
                            f"(PID/Port/Client检查失败)")
                # 清理失败的实例
                if server_instance and hasattr(server_instance, 'kill_server'):
                    try:
                        server_instance.kill_server()
                    except Exception as kill_e:
                        logging.error(f"清理失败服务器实例时出错: {model_name}: {kill_e}")
                return None
                
        except Exception as e:
            logging.error(f"创建VLLMServer时发生异常: {model_name}: {e}", exc_info=True)
            # 尝试清理部分创建的实例
            if server_instance and hasattr(server_instance, 'kill_server'):
                try:
                    server_instance.kill_server()
                except Exception as kill_e:
                    logging.error(f"异常处理时清理服务器失败: {model_name}: {kill_e}")
            return None

def idle_cleaner():
    """
    Thread function to remove models that haven't been accessed for more than IDLE_TIMEOUT.
    Only attempts to kill actual server instances, skipping placeholders or invalid entries.
    Handles potential race conditions during iteration and removal.
    """
    while True:
        time.sleep(60) # Check every 60 seconds
        with model_global_lock: # Acquire lock for safe access and modification
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

                # --- Check: Idle timeout for valid server instances ---
                if isinstance(server_instance, VLLMServer) and isinstance(last_access_time, datetime):
                    if (now - last_access_time) > IDLE_TIMEOUT:
                        logging.info(f"Model {m_name} idle timeout ({IDLE_TIMEOUT}) reached (Last access: {last_access_time}). Scheduling for removal.")
                        to_remove.append(m_name)

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
    global model_pool, model_global_lock
    logging.info("Starting background thread for inactive model cleanup.")
    while True:
        try: # Add error handling for the loop itself
            time.sleep(60 * 5) # Check every 5 minutes

            with model_global_lock: # Use the global lock to safely access pool
                now = datetime.now()
                inactive_models = []
                for model_name, entry in model_pool.items():
                    # Check only valid VLLMServer instances
                    if isinstance(entry["server"], VLLMServer):
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
    global model_pool, model_global_lock
    logging.info("Initiating VLLM server cleanup on application shutdown...")

    # Use the global lock to safely access and modify the pool
    with model_global_lock:
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
                    # Check if it's an actual VLLMServer instance
                    if isinstance(server_instance, VLLMServer):
                        logging.info(f"Shutting down VLLM server for model: {model_name}")
                        try:
                            # kill_server() should handle its own logging for PID etc.
                            server_instance.kill_server()
                        except Exception as e:
                            logging.error(f"Error shutting down server for {model_name}: {e}", exc_info=True)
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

def stop_model(model_name: str) -> dict:
    """
    手动停止并移除指定的模型。
    
    Args:
        model_name (str): 要停止的模型名称
        
    Returns:
        dict: 包含操作结果的字典，格式为 {'success': bool, 'message': str}
    """
    global model_pool, model_global_lock
    
    with model_global_lock:
        # 检查模型是否存在于池中
        if model_name not in model_pool:
            return {
                'success': False, 
                'message': f'模型 {model_name} 不存在于模型池中'
            }
        
        entry = model_pool[model_name]
        server_instance = entry.get("server")
        
        # 检查是否为有效的VLLMServer实例
        if not isinstance(server_instance, VLLMServer):
            # 移除无效的条目
            del model_pool[model_name]
            return {
                'success': False,
                'message': f'模型 {model_name} 条目无效，已从池中移除'
            }
        
        # 尝试停止服务器
        try:
            logging.info(f"手动停止模型服务器: {model_name}")
            server_instance.kill_server()
            
            # 从模型池中移除
            del model_pool[model_name]
            logging.info(f"成功停止并移除模型: {model_name}")
            
            return {
                'success': True,
                'message': f'成功停止模型 {model_name}'
            }
            
        except Exception as e:
            logging.error(f"停止模型 {model_name} 时发生错误: {e}", exc_info=True)
            
            # 即使停止失败，也尝试从池中移除
            try:
                if model_name in model_pool:
                    del model_pool[model_name]
            except KeyError:
                pass
                
            return {
                'success': False,
                'message': f'停止模型 {model_name} 时发生错误: {str(e)}'
            }


def list_active_models() -> dict:
    """
    获取当前活跃的模型列表。
    
    Returns:
        dict: 包含活跃模型信息的字典
    """
    global model_pool, model_global_lock
    
    with model_global_lock:
        active_models = []
        
        for model_name, entry in model_pool.items():
            server_instance = entry.get("server")
            last_access = entry.get("last_access")
            
            model_info = {
                'name': model_name,
                'status': 'ACTIVE' if isinstance(server_instance, VLLMServer) else 'INVALID',
                'last_access': last_access.isoformat() if isinstance(last_access, datetime) else str(last_access)
            }
            
            # 如果是VLLMServer实例，添加更多信息
            if isinstance(server_instance, VLLMServer):
                model_info.update({
                    'pid': server_instance.pid,
                    'port': server_instance.port,
                    'model_name': server_instance.model_name
                })
            
            active_models.append(model_info)
        
        return {
            'success': True,
            'models': active_models,
            'total_count': len(active_models)
        }


def stop_all_models() -> dict:
    """
    停止并移除所有活跃的模型。
    
    Returns:
        dict: 包含操作结果的字典，格式为 {'success': bool, 'message': str, 'stopped_models': list, 'failed_models': list}
    """
    global model_pool, model_global_lock
    
    with model_global_lock:
        # 如果模型池为空
        if not model_pool:
            return {
                'success': True,
                'message': 'Model pool is empty, no models to stop',
                'stopped_models': [],
                'failed_models': []
            }
        
        stopped_models = []
        failed_models = []
        
        # 获取当前所有模型的名称列表
        model_names = list(model_pool.keys())
        logging.info(f"开始停止所有模型，共 {len(model_names)} 个模型: {model_names}")
        
        for model_name in model_names:
            # 再次检查模型是否仍在池中（防止并发修改）
            if model_name not in model_pool:
                logging.warning(f"模型 {model_name} 在停止过程中从池中消失")
                continue
            
            entry = model_pool[model_name]
            server_instance = entry.get("server")
            
            # 检查是否为有效的VLLMServer实例
            if not isinstance(server_instance, VLLMServer):
                # 移除无效的条目
                try:
                    del model_pool[model_name]
                    failed_models.append({
                        'name': model_name,
                        'reason': 'Invalid entry, removed from pool'
                    })
                except KeyError:
                    pass
                continue
            
            # 尝试停止服务器
            try:
                logging.info(f"停止模型服务器: {model_name}")
                server_instance.kill_server()
                
                # 从模型池中移除
                del model_pool[model_name]
                stopped_models.append(model_name)
                logging.info(f"成功停止模型: {model_name}")
                
            except Exception as e:
                logging.error(f"停止模型 {model_name} 时发生错误: {e}", exc_info=True)
                
                # 即使停止失败，也尝试从池中移除
                try:
                    if model_name in model_pool:
                        del model_pool[model_name]
                except KeyError:
                    pass
                    
                failed_models.append({
                    'name': model_name,
                    'reason': str(e)
                })
        
        # 生成结果报告
        total_attempted = len(stopped_models) + len(failed_models)
        success = len(failed_models) == 0
        
        if success:
            message = f"Successfully stopped all {len(stopped_models)} models"
        else:
            message = f"Stopped {len(stopped_models)} models, {len(failed_models)} models failed to stop"
        
        logging.info(f"停止所有模型操作完成: {message}")
        
        return {
            'success': success,
            'message': message,
            'stopped_models': stopped_models,
            'failed_models': failed_models,
            'total_attempted': total_attempted
        }
