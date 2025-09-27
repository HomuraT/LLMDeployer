import threading
import time
import os
from datetime import datetime, timedelta
 
from typing import Dict, Any, Optional

from src.models.vllm_loader import VLLMServer
from src.utils.gpu_utils import find_available_gpu
from src.utils.log_config import logger
from src.utils.yaml_utils import YAMLConfigManager
from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH

# Assume these imports exist in your environment
# from src.models.vllm_loader import VLLMServer
# from src.models.gpu_utils import find_available_gpu

model_pool: Dict[str, Dict[str, Any]] = {}
# 全局锁：仅用于短时访问/修改 model_pool 的元数据，避免长时间阻塞
model_global_lock = threading.Lock()
IDLE_TIMEOUT = timedelta(minutes=60) # Example: 10 minutes idle timeout
MODEL_TTL_MINUTES = 60 # Time-to-live for inactive models

# 正在启动阶段已预留的 GPU 集合（避免并发选择相同 GPU 导致卡住）
reserved_gpus: set[int] = set()


def _read_tp_size_from_yaml(model_name: str) -> int:
    """
    从模型 YAML 读取 vllm.tensor_parallel_size，默认 1。
    兼容两种路径形式：带目录分隔的原始 `model_name` 与用下划线替换后的文件名。
    """
    try:
        # 形式一：目录结构
        yaml_path1 = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name + '.yaml')
        # 形式二：下划线扁平化
        yaml_path2 = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name.replace('/', '_') + '.yaml')
        yaml_path = yaml_path1 if os.path.exists(yaml_path1) else yaml_path2
        cfg = YAMLConfigManager.read_yaml(yaml_path) if os.path.exists(yaml_path) else None
        if cfg and isinstance(cfg, dict):
            v = cfg.get('vllm') or {}
            tp = v.get('tensor_parallel_size', 1)
            try:
                return int(tp)
            except Exception:
                return 1
        return 1
    except Exception:
        return 1


def _yaml_specifies_cuda(model_name: str) -> bool:
    """
    检测模型 YAML 是否通过 env.CUDA_VISIBLE_DEVICES 指定了 GPU。
    若已指定，则上层应跳过自动 GPU 预留/选择逻辑。
    """
    try:
        yaml_path1 = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name + '.yaml')
        yaml_path2 = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, model_name.replace('/', '_') + '.yaml')
        yaml_path = yaml_path1 if os.path.exists(yaml_path1) else yaml_path2
        if not os.path.exists(yaml_path):
            return False
        cfg = YAMLConfigManager.read_yaml(yaml_path)
        if not isinstance(cfg, dict):
            return False
        env_cfg = cfg.get('env')
        if not isinstance(env_cfg, dict):
            return False
        val = env_cfg.get('CUDA_VISIBLE_DEVICES')
        if val is None:
            return False
        # 空字符串也视为“未指定具体GPU”，因此需要非空判断
        try:
            return str(val).strip() != ''
        except Exception:
            return False
    except Exception:
        return False


def _reserve_gpus_for_start(model_name: str, tp_size: int, wait_timeout_sec: int = 180) -> Optional[list[int]]:
    """
    并发安全地为模型启动预留 tp_size 个 GPU；在超时时间内不断尝试。
    成功返回预留的 GPU 列表；失败返回 None。
    """
    deadline = time.time() + max(1, wait_timeout_sec)
    reserved_for_this: list[int] = []
    while time.time() < deadline and len(reserved_for_this) < tp_size:
        # 构造排除集合：已被其他启动任务预留的 GPU
        with model_global_lock:
            exclude = sorted(list(reserved_gpus))
        # 选择一个候选 GPU（find_available_gpu 返回按最大空闲的一组，取首个）
        candidates = find_available_gpu(model_name=model_name, exclude_gpus=exclude)
        if not candidates:
            time.sleep(1.0)
            continue
        chosen = candidates[0]
        # 尝试占用 chosen（双重检查）
        with model_global_lock:
            if chosen in reserved_gpus:
                # 已被他人抢占，重试
                pass
            else:
                reserved_gpus.add(chosen)
                reserved_for_this.append(chosen)
                logger.info(f"为模型 {model_name} 预留 GPU {chosen}（{len(reserved_for_this)}/{tp_size}）")
        if len(reserved_for_this) < tp_size:
            # 稍作等待再抢占下一个，避免忙等
            time.sleep(0.2)

    if len(reserved_for_this) == tp_size:
        return reserved_for_this

    # 失败则释放本次已预留
    if reserved_for_this:
        with model_global_lock:
            for g in reserved_for_this:
                reserved_gpus.discard(g)
        logger.warning(f"为模型 {model_name} 预留 GPU 超时/失败，已释放部分预留: {reserved_for_this}")
    return None


def _release_reserved_gpus(gpus: Optional[list[int]]):
    if not gpus:
        return
    with model_global_lock:
        for g in gpus:
            reserved_gpus.discard(g)
    logger.info(f"释放启动期预留的 GPU: {gpus}")

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

    # 先在锁内进行快速路径与占位，再在锁外执行耗时创建
    create_event: Optional[threading.Event] = None
    am_creator = False
    server_to_kill: Optional[VLLMServer] = None
    with model_global_lock:
        # 已存在条目：需要判断健康状态
        if model_name in model_pool:
            entry = model_pool[model_name]
            server_instance = entry.get("server")
            # 情况A：正在创建中 -> 等待
            if entry.get("creating") and isinstance(entry.get("event"), threading.Event):
                create_event = None
            # 情况B：有实例则直接复用（懒健康策略），失败由调用方处理
            elif isinstance(server_instance, VLLMServer):
                entry["last_access"] = datetime.now()
                logger.info(f"返回现有的服务器实例: {model_name}")
                return server_instance
            else:
                # 条目无效：转入创建路径
                logger.warning(f"模型 {model_name} 的池条目无效，重新创建")
                create_event = threading.Event()
                model_pool[model_name] = {
                    "creating": True,
                    "event": create_event,
                    "last_access": datetime.now(),
                    "pinned": False,
                }
                am_creator = True
        else:
            # 不存在：放置占位，开始创建
            logger.info(f"模型 {model_name} 不存在于池中，放置占位并开始创建新实例")
            create_event = threading.Event()
            model_pool[model_name] = {
                "creating": True,
                "event": create_event,
                "last_access": datetime.now(),
                "pinned": False,
            }
            am_creator = True

    if not am_creator:
        # 说明存在占位但不是我们创建，等待他人创建结果
        wait_event: Optional[threading.Event] = None
        with model_global_lock:
            entry = model_pool.get(model_name)
            if entry and entry.get("creating") and isinstance(entry.get("event"), threading.Event):
                wait_event = entry["event"]
        if wait_event:
            # 最多等待5分钟，避免永久卡死
            wait_event.wait(timeout=300)
        # 创建完成后读取结果
        with model_global_lock:
            entry = model_pool.get(model_name)
            if not entry:
                logger.error(f"模型 {model_name} 创建占位结束但条目缺失")
                return None
            server_instance = entry.get("server")
            if isinstance(server_instance, VLLMServer):
                entry["last_access"] = datetime.now()
                return server_instance
            logger.error(f"模型 {model_name} 创建失败或无效条目: {entry}")
            # 清理无效占位
            try:
                del model_pool[model_name]
            except KeyError:
                pass
            return None

    # 锁外：如果需要，先关闭失效的旧实例，避免GPU/端口占用导致重复启动
    if server_to_kill is not None:
        try:
            logger.info(f"Attempting to kill stale server before recreate: {model_name}")
            server_to_kill.kill_server()
        except Exception as e:
            logger.error(f"Error killing stale server for {model_name}: {e}")

    # 我们负责创建的路径：锁外执行耗时操作
    server_instance: VLLMServer | None = None
    try:
        # 若 YAML 已指定 CUDA_VISIBLE_DEVICES，则跳过自动 GPU 预留
        yaml_has_cuda = _yaml_specifies_cuda(model_name)
        tp_size = _read_tp_size_from_yaml(model_name)
        if tp_size < 1:
            tp_size = 1

        pre_reserved: Optional[list[int]] = None
        if not yaml_has_cuda:
            pre_reserved = _reserve_gpus_for_start(model_name, tp_size)
            if not pre_reserved:
                logger.error(f"未找到适合模型 {model_name} 的GPU，无法创建服务器")
                with model_global_lock:
                    model_pool[model_name] = {
                        "status": "FAILED",
                        "error": "NO_SUITABLE_GPU",
                        "last_access": datetime.now()
                    }
                return None
            logger.info(f"为模型 {model_name} 在GPU {pre_reserved} 上创建VLLMServer（按 tp_size={tp_size} 预留）")
        else:
            logger.info(f"检测到 YAML 指定 CUDA_VISIBLE_DEVICES，跳过自动 GPU 预留以使用 YAML 环境配置。")

        # 根据是否预留传入 cuda 参数（None 表示让子进程依 YAML/系统环境决定）
        server_instance = VLLMServer(model_name, cuda=pre_reserved)
        if (server_instance and server_instance.pid and server_instance.port and server_instance.client):
            with model_global_lock:
                existing_pinned = False
                try:
                    existing_pinned = bool(model_pool.get(model_name, {}).get("pinned", False))
                except Exception:
                    existing_pinned = False
                model_pool[model_name] = {
                    "server": server_instance,
                    "last_access": datetime.now(),
                    "last_healthy": datetime.now(),
                    "unhealthy_count": 0,
                    "pinned": existing_pinned,
                }
                # 通知等待者
                create_event.set()
            # 启动成功后释放预留（此时进程已占用显存，不再需要“软预留”）
            if pre_reserved:
                _release_reserved_gpus(pre_reserved)
            logger.info(f"VLLMServer创建成功: {model_name} (PID: {server_instance.pid}, Port: {server_instance.port})")
            return server_instance
        else:
            logger.error(f"VLLMServer初始化失败: {model_name} (PID/Port/Client检查失败)")
            if server_instance and hasattr(server_instance, 'kill_server'):
                try:
                    server_instance.kill_server()
                except Exception as kill_e:
                    logger.error(f"清理失败服务器实例时出错: {model_name}: {kill_e}")
            with model_global_lock:
                model_pool[model_name] = {
                    "status": "FAILED",
                    "error": "INIT_VALIDATION_FAILED",
                    "last_access": datetime.now()
                }
            if pre_reserved:
                _release_reserved_gpus(pre_reserved)
            return None
    except Exception as e:
        logger.error(f"创建VLLMServer时发生异常: {model_name}: {e}")
        if server_instance and hasattr(server_instance, 'kill_server'):
            try:
                server_instance.kill_server()
            except Exception as kill_e:
                logger.error(f"异常处理时清理服务器失败: {model_name}: {kill_e}")
        # 标记失败
        with model_global_lock:
            model_pool[model_name] = {
                "status": "FAILED",
                "error": str(e),
                "last_access": datetime.now()
            }
        # 异常同样释放预留
        try:
            _release_reserved_gpus(pre_reserved)
        except Exception:
            pass
        return None
    finally:
        # 无论成功失败，都应通知等待者结束等待
        try:
            create_event.set()
        except Exception:
            pass

def idle_cleaner():
    """
    Thread function to remove models that haven't been accessed for more than IDLE_TIMEOUT.
    Only attempts to kill actual server instances, skipping placeholders or invalid entries.
    Handles potential race conditions during iteration and removal.
    """
    while True:
        time.sleep(60)
        to_kill: list[tuple[str, VLLMServer]] = []
        with model_global_lock:
            now = datetime.now()
            current_model_names = list(model_pool.keys())
            for m_name in current_model_names:
                info = model_pool.get(m_name)
                if not isinstance(info, dict):
                    continue
                # 跳过正在创建的条目
                if info.get("creating"):
                    continue
                server_instance = info.get("server")
                last_access_time = info.get("last_access")
                # 跳过被固定的（不自动卸载）模型
                if info.get("pinned"):
                    continue
                if isinstance(server_instance, VLLMServer) and isinstance(last_access_time, datetime):
                    if (now - last_access_time) > IDLE_TIMEOUT:
                        # 从池中移除，并记录待杀
                        to_kill.append((m_name, server_instance))
                        try:
                            del model_pool[m_name]
                        except KeyError:
                            pass
        # 锁外执行 kill，避免长时间阻塞
        for m_name, server in to_kill:
            logger.info(f"Attempting to kill idle/stuck server for model {m_name}.")
            try:
                server.kill_server()
                logger.info(f"Successfully killed server for model {m_name}.")
            except Exception as e:
                logger.error(f"Error killing server for idle/stuck model {m_name}: {e}")

def cleanup_inactive_models():
    """
    Periodically checks the model pool and removes inactive models based on MODEL_TTL_MINUTES.
    Runs in a background thread.
    """
    global model_pool, model_global_lock
    logger.info("Starting background thread for inactive model cleanup.")
    while True:
        try: # Add error handling for the loop itself
            time.sleep(60 * 5) # Check every 5 minutes

            with model_global_lock: # Use the global lock to safely access pool
                now = datetime.now()
                inactive_models = []
                for model_name, entry in model_pool.items():
                    # 跳过被固定的（不自动卸载）模型
                    if entry.get("pinned"):
                        continue
                    # Check only valid VLLMServer instances
                    if isinstance(entry["server"], VLLMServer):
                        if now - entry["last_access"] > timedelta(minutes=MODEL_TTL_MINUTES):
                            inactive_models.append(model_name)

                if inactive_models:
                    logger.info(f"Found inactive models exceeding TTL ({MODEL_TTL_MINUTES} min): {inactive_models}")

                for model_name in inactive_models:
                    if model_name in model_pool: # Double-check existence before removal
                        entry = model_pool[model_name]
                        server_instance = entry["server"]
                        logger.info(f"Removing inactive model {model_name} due to TTL.")
                        # Ensure it's a valid server instance before killing
                        if hasattr(server_instance, 'kill_server') and callable(server_instance.kill_server):
                             try:
                                 server_instance.kill_server()
                             except Exception as e:
                                 logger.error(f"Error killing inactive server {model_name}: {e}")
                        else:
                            logger.warning(f"Attempted to remove inactive model {model_name}, but found unexpected server entry: {server_instance}")

                        # Remove from pool after attempting kill
                        del model_pool[model_name]
                        logger.info(f"Model {model_name} removed from pool.")
                    else:
                        logger.warning(f"Tried to remove inactive model {model_name}, but it was already gone.")

        except Exception as e:
            logger.error(f"Error in cleanup_inactive_models loop: {e}")
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
    logger.info("Initiating VLLM server cleanup on application shutdown...")

    to_kill: list[tuple[str, VLLMServer]] = []
    with model_global_lock:
        model_names = list(model_pool.keys())
        logger.info(f"Found models in pool to cleanup: {model_names}")
        for model_name in model_names:
            entry = model_pool.get(model_name)
            if not entry or not isinstance(entry, dict):
                # 清理异常条目
                try:
                    if model_name in model_pool:
                        del model_pool[model_name]
                except KeyError:
                    pass
                continue
            # 跳过正在创建的条目，但也从池中移除，避免泄漏
            if entry.get("creating"):
                try:
                    del model_pool[model_name]
                except KeyError:
                    pass
                continue
            server_instance = entry.get("server")
            if isinstance(server_instance, VLLMServer):
                to_kill.append((model_name, server_instance))
            # 无论是否有效，都从池中移除
            try:
                del model_pool[model_name]
            except KeyError:
                pass
    # 锁外执行 kill
    for model_name, server_instance in to_kill:
        logger.info(f"Shutting down VLLM server for model: {model_name}")
        try:
            server_instance.kill_server()
        except Exception as e:
            logger.error(f"Error shutting down server for {model_name}: {e}")

    logger.info("VLLM server global cleanup finished.")

def stop_model(model_name: str) -> dict:
    """
    手动停止并移除指定的模型。
    
    Args:
        model_name (str): 要停止的模型名称
        
    Returns:
        dict: 包含操作结果的字典，格式为 {'success': bool, 'message': str}
    """
    global model_pool, model_global_lock
    
    server_to_kill: Optional[VLLMServer] = None
    with model_global_lock:
        if model_name not in model_pool:
            return {'success': False, 'message': f'模型 {model_name} 不存在于模型池中'}
        entry = model_pool.get(model_name)
        if not isinstance(entry, dict):
            try:
                del model_pool[model_name]
            except KeyError:
                pass
            return {'success': False, 'message': f'模型 {model_name} 条目无效，已从池中移除'}
        # 正在创建的直接移除
        if entry.get("creating"):
            try:
                del model_pool[model_name]
            except KeyError:
                pass
            return {'success': True, 'message': f'成功停止模型 {model_name} (创建中占位已移除)'}
        server_instance = entry.get("server")
        if isinstance(server_instance, VLLMServer):
            server_to_kill = server_instance
            try:
                del model_pool[model_name]
            except KeyError:
                pass
        else:
            try:
                del model_pool[model_name]
            except KeyError:
                pass
            return {'success': False, 'message': f'模型 {model_name} 条目无效，已从池中移除'}

    # 锁外执行 kill
    if server_to_kill:
        try:
            logger.info(f"手动停止模型服务器: {model_name}")
            server_to_kill.kill_server()
            logger.info(f"成功停止并移除模型: {model_name}")
            return {'success': True, 'message': f'成功停止模型 {model_name}'}
        except Exception as e:
            logger.error(f"停止模型 {model_name} 时发生错误: {e}")
            return {'success': False, 'message': f'停止模型 {model_name} 时发生错误: {str(e)}'}
    return {'success': False, 'message': f'模型 {model_name} 未找到可停止的实例'}


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
            if not isinstance(entry, dict):
                active_models.append({'name': model_name, 'status': 'INVALID'})
                continue
            last_access = entry.get("last_access")
            if entry.get("creating"):
                active_models.append({
                    'name': model_name,
                    'status': 'STARTING',
                    'last_access': last_access.isoformat() if isinstance(last_access, datetime) else str(last_access),
                    'pinned': bool(entry.get('pinned', False)),
                })
                continue
            if entry.get("status") == "FAILED":
                active_models.append({
                    'name': model_name,
                    'status': 'FAILED',
                    'error': entry.get('error'),
                    'last_access': last_access.isoformat() if isinstance(last_access, datetime) else str(last_access),
                    'pinned': bool(entry.get('pinned', False)),
                })
                continue
            server_instance = entry.get("server")
            model_info = {
                'name': model_name,
                'status': 'STARTED' if isinstance(server_instance, VLLMServer) else 'INVALID',
                'last_access': last_access.isoformat() if isinstance(last_access, datetime) else str(last_access)
            }
            # 暴露 pinned 状态
            try:
                model_info['pinned'] = bool(entry.get('pinned', False))
            except Exception:
                model_info['pinned'] = False
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


def set_model_pinned(model_name: str, pinned: bool) -> dict:
    """
    设置指定模型的保留标记（pinned）。仅对模型池中的现有条目生效。

    Args:
        model_name: 模型名称
        pinned: 是否保留（True=不被自动卸载）

    Returns:
        dict: {success: bool, message: str, name: str, pinned: bool}
    """
    global model_pool, model_global_lock

    with model_global_lock:
        entry = model_pool.get(model_name)
        if not entry or not isinstance(entry, dict):
            return {'success': False, 'message': f'模型 {model_name} 不存在', 'name': model_name, 'pinned': False}
        try:
            entry['pinned'] = bool(pinned)
            return {'success': True, 'message': 'OK', 'name': model_name, 'pinned': entry['pinned']}
        except Exception as e:
            return {'success': False, 'message': str(e), 'name': model_name, 'pinned': bool(entry.get('pinned', False))}


def stop_all_models() -> dict:
    """
    停止并移除所有活跃的模型。
    
    Returns:
        dict: 包含操作结果的字典，格式为 {'success': bool, 'message': str, 'stopped_models': list, 'failed_models': list}
    """
    global model_pool, model_global_lock
    
    stopped_models: list[str] = []
    failed_models: list[dict] = []
    to_kill: list[tuple[str, VLLMServer]] = []
    with model_global_lock:
        if not model_pool:
            return {'success': True, 'message': 'Model pool is empty, no models to stop', 'stopped_models': [], 'failed_models': []}
        model_names = list(model_pool.keys())
        logger.info(f"开始停止所有模型，共 {len(model_names)} 个模型: {model_names}")
        for model_name in model_names:
            entry = model_pool.get(model_name)
            if not entry or not isinstance(entry, dict):
                try:
                    del model_pool[model_name]
                except KeyError:
                    pass
                failed_models.append({'name': model_name, 'reason': 'Invalid entry, removed from pool'})
                continue
            if entry.get("creating"):
                try:
                    del model_pool[model_name]
                except KeyError:
                    pass
                stopped_models.append(model_name)
                continue
            server_instance = entry.get("server")
            if isinstance(server_instance, VLLMServer):
                to_kill.append((model_name, server_instance))
                try:
                    del model_pool[model_name]
                except KeyError:
                    pass
            else:
                try:
                    del model_pool[model_name]
                except KeyError:
                    pass
                failed_models.append({'name': model_name, 'reason': 'Invalid entry, removed from pool'})
    # 锁外执行 kill
    for model_name, server_instance in to_kill:
        try:
            logger.info(f"停止模型服务器: {model_name}")
            server_instance.kill_server()
            stopped_models.append(model_name)
            logger.info(f"成功停止模型: {model_name}")
        except Exception as e:
            logger.error(f"停止模型 {model_name} 时发生错误: {e}")
            failed_models.append({'name': model_name, 'reason': str(e)})

    total_attempted = len(stopped_models) + len(failed_models)
    success = len(failed_models) == 0
    if success:
        message = f"Successfully stopped all {len(stopped_models)} models"
    else:
        message = f"Stopped {len(stopped_models)} models, {len(failed_models)} models failed to stop"
    logger.info(f"停止所有模型操作完成: {message}")
    return {'success': success, 'message': message, 'stopped_models': stopped_models, 'failed_models': failed_models, 'total_attempted': total_attempted}
