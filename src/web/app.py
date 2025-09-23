import argparse
import os
import threading
import time # Added for potential sleep/retry logic if needed

import requests
from flask import Flask, request, Response, jsonify, render_template, redirect, url_for
# from pandas.tests.io.formats.test_to_html import justify # Removed unused import
from rich.console import Console

from src.web.multi_model_utils import get_or_create_model, idle_cleaner, stop_model, list_active_models, stop_all_models
from src.models.vllm_loader import VLLMServer # Import VLLMServer for type hinting and error handling
from src.utils.log_config import logger
from src.utils.gpu_utils import get_gpu_stats
from src.utils.system_utils import get_system_stats
from src.utils.model_registry import list_available_models
from src.utils.modelscope_utils import (
    search_modelscope_models,
    download_modelscope_model,
    write_vllm_yaml_for_model,
    DownloadTaskStore,
    parse_modelscope_url_to_id,
)

app = Flask(__name__)
console = Console()

requests_time_out = 1200

# In-memory tasks for ModelScope downloads
ms_tasks = DownloadTaskStore()


@app.route('/')
def index():
    return redirect(url_for('admin_page'))


@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/generate', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    """
    Handles incoming chat completion requests, forwards them to the appropriate
    vLLM server instance, and handles connection errors by triggering a server restart.
    """
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({"error": "Missing 'model' in request JSON"}), 400
        # logging.info('收到消息：'+str(data))
        model_name = data['model']
    except Exception as e:
        logger.error(f"Error parsing request JSON: {e}")
        return jsonify({"error": "Invalid JSON request"}), 400

    try:
        llm: VLLMServer = get_or_create_model(model_name)
        if not llm or not llm.port:
             # This might happen if the initial startup in VLLMServer failed
             logger.error(f"Failed to get or create a valid server instance for model {model_name}.")
             return jsonify({"error": f"Service for model {model_name} is unavailable or failed to start."}), 503

        forward_url = f"http://127.0.0.1:{llm.port}/v1/chat/completions"
        is_stream = data.get('stream', False)

        try:
            if is_stream:
                # 第一次尝试（禁用代理）
                forward_response = requests.post(
                    forward_url,
                    json=data,
                    stream=True,
                    timeout=requests_time_out,
                    proxies={"http": None, "https": None},
                )
                forward_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                def event_stream():
                    try:
                        for chunk in forward_response.iter_content(chunk_size=None):
                            if chunk:
                                yield chunk
                    except requests.exceptions.ChunkedEncodingError as stream_err:
                         logger.error(f"Stream interrupted for {model_name}: {stream_err}")
                         # Client likely disconnected, nothing specific to yield here
                    except Exception as gen_err:
                         logger.error(f"Error during stream generation for {model_name}: {gen_err}")
                         yield jsonify({"error": "Error during stream generation"}).data # Try to inform client
                    finally:
                         forward_response.close() # Ensure connection is closed

                # Check content type, VLLM usually uses text/event-stream
                content_type = forward_response.headers.get('content-type', 'application/json')
                return Response(event_stream(), content_type=content_type)
            else:
                try:
                    # 第一次尝试（禁用代理）
                    forward_response = requests.post(
                        forward_url,
                        json=data,
                        timeout=requests_time_out,
                        proxies={"http": None, "https": None},
                    )
                    forward_response.raise_for_status()
                except requests.exceptions.ConnectionError:
                    # 小退避后重试一次，避免瞬时抖动误判为服务挂掉
                    time.sleep(0.2)
                    forward_response = requests.post(
                        forward_url,
                        json=data,
                        timeout=requests_time_out,
                        proxies={"http": None, "https": None},
                    )
                    forward_response.raise_for_status()
                # Forward the exact response from the vLLM server
                return forward_response.content, forward_response.status_code, forward_response.headers.items()

        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error to VLLM server {model_name} at {forward_url}: {conn_err}")
            # 先进行快速健康检查，避免误判导致重启
            health_url = f"http://127.0.0.1:{llm.port}/health"
            healthy = False
            for _ in range(3):
                try:
                    hr = requests.get(health_url, timeout=1.0, proxies={"http": None, "https": None})
                    if hr.status_code == 200:
                        healthy = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.1)
            if healthy:
                # 健康则认为是瞬时抖动，退避后重试一次
                time.sleep(0.2)
                try:
                    forward_response = requests.post(
                        forward_url,
                        json=data,
                        timeout=requests_time_out,
                        proxies={"http": None, "https": None},
                    )
                    forward_response.raise_for_status()
                    return forward_response.content, forward_response.status_code, forward_response.headers.items()
                except Exception:
                    pass
            # 健康检查失败：触发重启
            llm.handle_connection_error()
            return jsonify({"error": f"Service for model {model_name} is temporarily unavailable due to connection issue. Restart initiated. Please try again shortly."}), 503
        except requests.exceptions.Timeout as timeout_err:
             logger.error(f"Timeout connecting to VLLM server {model_name} at {forward_url}: {timeout_err}")
             # Decide if timeout should also trigger restart, or just indicate temporary issue
             # llm.handle_connection_error() # Optional: uncomment if timeout implies server death
             return jsonify({"error": f"Request timed out connecting to model {model_name}. The service might be overloaded or unresponsive."}), 504 # Gateway Timeout
        except requests.exceptions.RequestException as req_err:
            # Catch other request errors (like HTTPError from raise_for_status)
            logger.error(f"Error forwarding request to VLLM server {model_name} at {forward_url}: {req_err}")
            status_code = forward_response.status_code if 'forward_response' in locals() and hasattr(forward_response, 'status_code') else 502 # Bad Gateway
            error_content = forward_response.text if 'forward_response' in locals() and hasattr(forward_response, 'text') else str(req_err)
            # Avoid returning potentially large/sensitive internal error details directly
            return jsonify({"error": f"Failed to get response from model {model_name}. Service returned status {status_code}."}), status_code

    except Exception as e:
        # Catch-all for unexpected errors during model getting/creation or request handling
        logger.error(f"Unexpected error handling request for model {model_name}: {e}")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500


@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """
    Handles incoming embedding requests, forwards them to the appropriate
    vLLM server instance's /v1/embeddings endpoint, and handles connection errors.

    Input:
        JSON request body conforming to OpenAI embeddings API:
        {
            "model": "model_name",
            "input": "text" or ["list", "of", "texts"],
            ... (other optional parameters like encoding_format)
        }
    Output:
        JSON response from the vLLM server or an error JSON.
    """
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({"error": "Missing 'model' in request JSON"}), 400
        model_name = data['model']
        if 'input' not in data:
             return jsonify({"error": "Missing 'input' in request JSON"}), 400
        # logger.info(f"Received embedding request for model {model_name}")
    except Exception as e:
        logger.error(f"Error parsing embedding request JSON: {e}")
        return jsonify({"error": "Invalid JSON request"}), 400

    try:
        llm: VLLMServer = get_or_create_model(model_name)
        if not llm or not llm.port:
             logger.error(f"Failed to get or create a valid server instance for embedding model {model_name}.")
             return jsonify({"error": f"Service for embedding model {model_name} is unavailable or failed to start."}), 503

        forward_url = f"http://127.0.0.1:{llm.port}/v1/embeddings"

        try:
            # Forward the request to the vLLM server's embedding endpoint（禁用代理，并在瞬时失败时重试一次）
            try:
                forward_response = requests.post(
                    forward_url,
                    json=data,
                    timeout=120,
                    proxies={"http": None, "https": None},
                )
                forward_response.raise_for_status()
            except requests.exceptions.ConnectionError:
                time.sleep(0.2)
                forward_response = requests.post(
                    forward_url,
                    json=data,
                    timeout=120,
                    proxies={"http": None, "https": None},
                )
                forward_response.raise_for_status()

            # Forward the exact response from the vLLM server
            # Note: vLLM embedding response content-type is typically application/json
            return forward_response.content, forward_response.status_code, forward_response.headers.items()

        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error to VLLM server {model_name} (embeddings) at {forward_url}: {conn_err}")
            # 先健康检查，避免误杀
            health_url = f"http://127.0.0.1:{llm.port}/health"
            healthy = False
            for _ in range(3):
                try:
                    hr = requests.get(health_url, timeout=1.0, proxies={"http": None, "https": None})
                    if hr.status_code == 200:
                        healthy = True
                        break
                except requests.exceptions.RequestException:
                    time.sleep(0.1)
            if healthy:
                time.sleep(0.2)
                try:
                    forward_response = requests.post(
                        forward_url,
                        json=data,
                        timeout=120,
                        proxies={"http": None, "https": None},
                    )
                    forward_response.raise_for_status()
                    return forward_response.content, forward_response.status_code, forward_response.headers.items()
                except Exception:
                    pass
            # 健康检查失败：触发重启
            llm.handle_connection_error() # Trigger restart mechanism
            return jsonify({"error": f"Service for embedding model {model_name} is temporarily unavailable due to connection issue. Restart initiated. Please try again shortly."}), 503
        except requests.exceptions.Timeout as timeout_err:
             logger.error(f"Timeout connecting to VLLM server {model_name} (embeddings) at {forward_url}: {timeout_err}")
             # Decide if timeout should also trigger restart, or just indicate temporary issue
             # llm.handle_connection_error() # Optional: uncomment if timeout implies server death
             return jsonify({"error": f"Request timed out connecting to embedding model {model_name}. The service might be overloaded or unresponsive."}), 504 # Gateway Timeout
        except requests.exceptions.RequestException as req_err:
            # Catch other request errors (like HTTPError from raise_for_status)
            logger.error(f"Error forwarding embedding request to VLLM server {model_name} at {forward_url}: {req_err}")
            status_code = forward_response.status_code if 'forward_response' in locals() and hasattr(forward_response, 'status_code') else 502 # Bad Gateway
            error_content = forward_response.text if 'forward_response' in locals() and hasattr(forward_response, 'text') else str(req_err)
            # Avoid returning potentially large/sensitive internal error details directly
            return jsonify({"error": f"Failed to get embedding response from model {model_name}. Service returned status {status_code}."}), status_code

    except Exception as e:
        # Catch-all for unexpected errors during model getting/creation or request handling
        logger.error(f"Unexpected error handling embedding request for model {model_name}: {e}")
        return jsonify({"error": "An unexpected internal server error occurred during embedding request."}), 500


@app.route('/v1/stop_model', methods=['POST'])
def stop_model_endpoint():
    """
    停止并卸载指定的vLLM模型。
    
    输入:
        JSON请求体: {'model': 'model_name'}
    输出:
        JSON响应，包含操作结果
    """
    try:
        data = request.json
        if not data or 'model' not in data:
            return jsonify({
                "success": False,
                "error": "请求JSON中缺少'model'字段"
            }), 400
            
        model_name = data['model']
        logger.info(f"收到停止模型请求: {model_name}")
        
    except Exception as e:
        logger.error(f"解析停止模型请求JSON时出错: {e}")
        return jsonify({
            "success": False,
            "error": "无效的JSON请求"
        }), 400
    
    try:
        # 调用停止模型函数
        result = stop_model(model_name)
        
        if result['success']:
            logger.info(f"成功停止模型: {model_name}")
            return jsonify(result), 200
        else:
            logger.warning(f"停止模型失败: {model_name}, 原因: {result['message']}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"停止模型 {model_name} 时发生意外错误: {e}")
        return jsonify({
            "success": False,
            "error": f"停止模型时发生内部服务器错误: {str(e)}"
        }), 500


@app.route('/v1/models/active', methods=['GET'])
def list_active_models_endpoint():
    """
    获取当前活跃的模型列表。
    
    输入: 无
    输出:
        JSON响应，包含活跃模型的列表和相关信息
    """
    try:
        result = list_active_models()
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"获取活跃模型列表时发生错误: {e}")
        return jsonify({
            "success": False,
            "error": f"获取模型列表时发生内部服务器错误: {str(e)}"
        }), 500


@app.route('/v1/stop_all_models', methods=['POST'])
def stop_all_models_endpoint():
    """
    停止并卸载所有活跃的vLLM模型。
    
    输入: 无需输入参数（POST请求）
    输出:
        JSON响应，包含操作结果，包括成功停止的模型列表和失败的模型列表
        {
            "success": bool,
            "message": str,
            "stopped_models": list,
            "failed_models": list,
            "total_attempted": int
        }
    """
    try:
        logger.info("Received request to stop all models")
        
        # 调用停止所有模型的函数
        result = stop_all_models()
        
        if result['success']:
            logger.info(f"Successfully stopped all models: {result['message']}")
            return jsonify(result), 200
        else:
            logger.warning(f"Partially failed to stop all models: {result['message']}")
            return jsonify(result), 207  # 207 Multi-Status，表示部分成功
            
    except Exception as e:
        logger.error(f"Unexpected error while stopping all models: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error while stopping all models: {str(e)}",
            "stopped_models": [],
            "failed_models": [],
            "total_attempted": 0
        }), 500


# --- Admin APIs ---
@app.route('/api/admin/gpu', methods=['GET'])
def admin_gpu_stats():
    try:
        return jsonify(get_gpu_stats()), 200
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/system', methods=['GET'])
def admin_system_stats():
    try:
        return jsonify({"success": True, **get_system_stats()}), 200
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/models', methods=['GET'])
def admin_models():
    try:
        return jsonify(list_active_models()), 200
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/stop_model', methods=['POST'])
def admin_stop_model():
    data = request.get_json(silent=True) or {}
    model_name = data.get('model')
    if not model_name:
        return jsonify({"success": False, "message": "Missing 'model'"}), 400
    try:
        result = stop_model(model_name)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        logger.error(f"Error stopping model {model_name}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/stop_all', methods=['POST'])
def admin_stop_all():
    try:
        result = stop_all_models()
        return jsonify(result), (200 if result.get('success') else 207)
    except Exception as e:
        logger.error(f"Error stopping all models: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/models/available', methods=['GET'])
def admin_list_available_models():
    try:
        models = list_available_models()
        return jsonify({"success": True, "models": models, "total": len(models)}), 200
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/models/start', methods=['POST'])
def admin_start_model():
    try:
        data = request.get_json(silent=True) or {}
        model_id = data.get('model')
        if not model_id:
            return jsonify({"success": False, "message": "Missing 'model'"}), 400
        # 异步创建：立刻放置占位并返回 STARTING，由后台线程启动
        def _bg_start():
            try:
                get_or_create_model(model_id)
            except Exception:
                pass
        threading.Thread(target=_bg_start, daemon=True).start()
        return jsonify({"success": True, "model": model_id, "status": "STARTING"}), 202
    except Exception as e:
        logger.error(f"Error starting model: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


# --- ModelScope Admin APIs ---
@app.route('/api/admin/modelscope/search', methods=['GET'])
def admin_modelscope_search():
    try:
        kw = (request.args.get('q') or request.args.get('kw') or '').strip()
        page = int(request.args.get('page') or 1)
        size = int(request.args.get('size') or 20)
        items = search_modelscope_models(kw, page=page, size=size)
        return jsonify({"success": True, "items": items, "total": len(items)}), 200
    except Exception as e:
        logger.error(f"Error searching ModelScope: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/modelscope/download', methods=['POST'])
def admin_modelscope_download():
    try:
        data = request.get_json(silent=True) or {}
        model_id = data.get('model_id') or data.get('model')
        dest_dir = data.get('dest_dir') or data.get('path')
        tp_size = int(data.get('tensor_parallel_size') or data.get('tp_size') or 1)
        gmu = data.get('gpu_memory_utilization')
        if gmu is not None:
            try:
                gmu = float(gmu)
            except Exception:
                gmu = None
        if not model_id:
            return jsonify({"success": False, "message": "Missing 'model_id'"}), 400

        import time as _time
        task_id = f"{model_id}|{int(_time.time())}"
        ms_tasks.create(task_id, {
            'id': task_id,
            'model_id': model_id,
            'dest_dir': dest_dir,
            'status': 'PENDING',
            'message': '',
            'yaml_path': None,
            'local_dir': None,
        })

        def _bg_download():
            try:
                ms_tasks.update(task_id, status='DOWNLOADING')
                local_dir = download_modelscope_model(model_id, dest_dir)
                ms_tasks.update(task_id, status='WRITING_YAML', local_dir=local_dir)
                yaml_path = write_vllm_yaml_for_model(model_id, local_dir, tensor_parallel_size=tp_size, gpu_memory_utilization=gmu)
                ms_tasks.update(task_id, status='DONE', yaml_path=yaml_path)
                logger.info(f"ModelScope 下载完成并生成配置: {model_id} -> {yaml_path}")
            except Exception as e:
                logger.error(f"ModelScope 下载/生成配置失败: {model_id}: {e}")
                ms_tasks.update(task_id, status='FAILED', message=str(e))

        threading.Thread(target=_bg_download, daemon=True).start()
        return jsonify({"success": True, "task_id": task_id, "status": "STARTED"}), 202
    except Exception as e:
        logger.error(f"Error starting ModelScope download: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/modelscope/tasks', methods=['GET'])
def admin_modelscope_tasks():
    try:
        return jsonify({"success": True, "tasks": list(ms_tasks.all().values())}), 200
    except Exception as e:
        logger.error(f"Error getting ModelScope tasks: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/modelscope/write_yaml', methods=['POST'])
def admin_modelscope_write_yaml():
    """
    输入: JSON { model/model_id/url, local_dir, tensor_parallel_size?: int, gpu_memory_utilization?: float }
    输出: { success, yaml_path }
    作用: 根据本地已下载目录生成/覆盖 vLLM YAML 配置。
    """
    try:
        data = request.get_json(silent=True) or {}
        url_or_id = (data.get('model') or data.get('model_id') or data.get('url') or '').strip()
        local_dir = data.get('local_dir') or data.get('path')
        if not url_or_id:
            return jsonify({"success": False, "message": "缺少 model/model_id/url"}), 400
        if not local_dir:
            return jsonify({"success": False, "message": "缺少 local_dir"}), 400
        model_id = parse_modelscope_url_to_id(url_or_id) or url_or_id
        tp_size = int(data.get('tensor_parallel_size') or data.get('tp_size') or 1)
        gmu = data.get('gpu_memory_utilization')
        try:
            gmu = None if gmu is None else float(gmu)
        except Exception:
            gmu = None
        yaml_path = write_vllm_yaml_for_model(model_id, local_dir, tensor_parallel_size=tp_size, gpu_memory_utilization=gmu)
        return jsonify({"success": True, "yaml_path": yaml_path}), 200
    except Exception as e:
        logger.error(f"Error writing YAML for ModelScope: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/admin/modelscope/prepare_from_url', methods=['POST'])
def admin_modelscope_prepare_from_url():
    """
    输入: JSON { url: str, dest_dir?: str, tensor_parallel_size?: int, gpu_memory_utilization?: float }
    输出: 生成的下载脚本字符串(script)、解析出的 model_id、预期 YAML 路径。

    说明: 该接口不直接触发下载，仅生成脚本，便于在受限环境中手动或外部系统执行。
    """
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get('url') or '').strip()
        dest_dir = data.get('dest_dir') or data.get('path') or '/datanfs4/renlin24/file/models/modelscope'
        tp_size = int(data.get('tensor_parallel_size') or data.get('tp_size') or 1)
        gmu = data.get('gpu_memory_utilization')
        try:
            gmu = None if gmu is None else float(gmu)
        except Exception:
            gmu = None
        model_id = parse_modelscope_url_to_id(url) or ''
        if not model_id:
            return jsonify({"success": False, "message": "无法从URL解析出ModelScope的 model_id"}), 400

        # 预估 YAML 路径
        yaml_rel = model_id + '.yaml'
        # 与 compute_yaml_path_for_model 一致的路径前缀
        from src.utils.config_utils import VLLM_MODEL_CONFIG_BASE_PATH
        yaml_path = os.path.join(VLLM_MODEL_CONFIG_BASE_PATH, yaml_rel)

        # 返回单行命令：使用 ModelScope CLI 下载到包含 Org/Repo 的目录
        # 形如: <dest_dir>/models/Org/Repo
        full_local_dir = os.path.join(dest_dir, 'models', model_id)
        script = f"modelscope download --model \"{model_id}\" --local_dir \"{full_local_dir}\""

        return jsonify({
            "success": True,
            "model_id": model_id,
            "yaml_path": yaml_path,
            "script": script,
        }), 200
    except Exception as e:
        logger.error(f"Error preparing script from URL: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


def run():
    """CLI entrypoint to run the Flask App."""
    parser = argparse.ArgumentParser(description="Run text-generation model with specified parameters.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the Flask app.')
    parser.add_argument('--port', type=int, default=5824, help='Port for the Flask app.')
    args = parser.parse_args()

    # Start usage-monitor in a thread (daemon = True)
    threading.Thread(target=idle_cleaner, daemon=True).start()

    app.run(host=args.host, port=args.port)
