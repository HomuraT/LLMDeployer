import os
from typing import List, Dict

from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH


def list_available_models() -> List[Dict[str, str]]:
    """
    递归扫描 VLLM 模型配置目录，返回可用模型列表。

    返回示例：[{"id": "Qwen/Qwen2.5-7B-Instruct", "path": "/abs/.../Qwen/Qwen2.5-7B-Instruct.yaml"}]
    """
    base = os.path.abspath(VLLM_MODEL_CONFIG_BASE_PATH)
    results: List[Dict[str, str]] = []
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith('.yaml'):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base)
            model_id = rel_path[:-5] if rel_path.lower().endswith('.yaml') else rel_path
            model_id = model_id.replace('\\', '/')
            results.append({
                'id': model_id,
                'path': full_path,
                'name': os.path.basename(model_id)
            })
    # 排序：按目录/文件名
    results.sort(key=lambda x: x['id'].lower())
    return results


