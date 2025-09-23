import os
import threading
from typing import Any, Dict, List, Optional

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
from src.utils.yaml_utils import YAMLConfigManager
from src.utils.log_config import logger
import requests
from urllib.parse import urlparse


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error(f"创建目录失败: {path}: {e}")
        raise


def search_modelscope_models(keyword: str, page: int = 1, size: int = 20) -> List[Dict[str, Any]]:
    """
    仅使用 ModelScope Dolphin API 搜索模型。

    返回规范化字段: id, name, tasks(可选), framework(可选), downloads(可选)
    """
    kw = (keyword or '').strip()
    if not kw:
        return []
    url = "https://www.modelscope.cn/api/v1/dolphin/models"
    payload = {
        "page": max(1, int(page or 1)),
        "pageSize": max(1, min(int(size or 20), 100)),
        "Filters": {
            "Name": kw,
        }
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/plain, */*',
        'User-Agent': 'LLMDeployer/ModelScopeSearch',
    }
    try:
        r = requests.put(url, json=payload, timeout=20, headers=headers, proxies={"http": None, "https": None})
        if r.status_code != 200:
            logger.warning(f"官方 API 搜索失败: HTTP {r.status_code}")
            return []
        data = r.json() if r.content else {}
        models = (((data or {}).get('Data') or {}).get('Model') or {}).get('Models') or []
        results: List[Dict[str, Any]] = []
        for it in models:
            unique = it.get('UniqueName') or it.get('Name')
            mid = None
            if isinstance(unique, str) and '/' in unique:
                mid = unique
            else:
                owner = it.get('CreatedBy') or it.get('Owner') or ''
                name = it.get('Name') or ''
                if owner and name:
                    mid = f"{owner}/{name}"
            if not mid:
                continue
            results.append({
                "id": mid,
                "name": it.get('ChineseName') or it.get('Name') or mid.split('/')[-1],
                "tasks": None,
                "framework": ",".join(it.get('Frameworks') or []) if isinstance(it.get('Frameworks'), list) else None,
                "downloads": it.get('Downloads'),
            })
        return results
    except Exception as e:
        logger.warning(f"调用官方 API 失败: {e}")
        return []


def download_modelscope_model(model_id: str, dest_dir: Optional[str] = None) -> str:
    """
    下载指定 ModelScope 模型，返回本地模型目录。

    优先使用 cache_dir，将模型缓存到指定目录；返回值为实际落地目录（由 SDK 决定）。
    """
    if not model_id or "/" not in model_id:
        raise ValueError("model_id 需要形如 'Org/Repo' 的格式")
    try:
        from modelscope import snapshot_download
    except Exception as e:
        logger.error(f"未安装或无法导入 modelscope: {e}")
        raise

    cache_dir = None
    if dest_dir:
        cache_dir = os.path.abspath(dest_dir)
        ensure_dir(cache_dir)
    try:
        local_path = snapshot_download(model_id, cache_dir=cache_dir)
        if not local_path or not os.path.isdir(local_path):
            raise RuntimeError(f"snapshot_download 返回无效目录: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"下载模型 {model_id} 失败: {e}")
        raise


def parse_modelscope_url_to_id(url_or_id: str) -> Optional[str]:
    """
    将 ModelScope 模型页面 URL 解析为 `Org/Repo` 形式的 model_id。

    支持示例:
    - https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct
    - https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct/summary
    - 直接传入 "Qwen/Qwen3-Next-80B-A3B-Instruct"
    """
    if not url_or_id:
        return None
    s = url_or_id.strip().strip("/")
    # 已经是 Org/Repo 形式
    if "/" in s and not s.lower().startswith("http"):
        parts = s.split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return f"{parts[0]}/{parts[1]}"
        return None

    # URL 解析
    try:
        p = urlparse(s)
        if not p.netloc:
            return None
        host = p.netloc.lower()
        if not (host.endswith("modelscope.cn") or host.endswith("modelscope.com")):
            return None
        path = p.path.strip("/")
        # 期望 path: models/Org/Repo/(optional extra)
        seg = path.split("/")
        if len(seg) >= 3 and seg[0].lower() == "models":
            org = seg[1]
            repo = seg[2]
            if org and repo:
                return f"{org}/{repo}"
        return None
    except Exception:
        return None


def compute_yaml_path_for_model(model_id: str) -> str:
    """
    计算 YAML 配置路径，例如: resources/vllm_models/Qwen/Qwen3-8B.yaml
    """
    safe_rel = model_id.replace("\\", "/").strip("/")
    yaml_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, f"{safe_rel}.yaml")
    ensure_dir(os.path.dirname(yaml_path))
    return yaml_path


def write_vllm_yaml_for_model(model_id: str, local_model_dir: str, tensor_parallel_size: int = 1, gpu_memory_utilization: Optional[float] = None) -> str:
    """
    生成 vLLM YAML 配置文件，返回 YAML 路径。
    vllm.model 指向本地目录；served-model-name 保留原始 model_id。
    """
    cfg: Dict[str, Any] = {
        "vllm": {
            "model": os.path.abspath(local_model_dir),
            "served-model-name": model_id,
            "tensor_parallel_size": int(tensor_parallel_size) if tensor_parallel_size and tensor_parallel_size > 0 else 1,
        }
    }
    if isinstance(gpu_memory_utilization, (int, float)):
        cfg["vllm"]["gpu_memory_utilization"] = float(gpu_memory_utilization)

    yaml_path = compute_yaml_path_for_model(model_id)
    YAMLConfigManager.write_yaml(yaml_path, cfg)
    logger.info(f"已生成 vLLM 配置: {yaml_path}")
    return yaml_path


# 简单的下载任务管理（内存级）
class DownloadTaskStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, task_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = payload

    def update(self, task_id: str, **kwargs) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._tasks)