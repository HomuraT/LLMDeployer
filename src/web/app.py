import argparse
import threading
import time # Added for potential sleep/retry logic if needed

import requests
from flask import Flask, request, Response, jsonify
# from pandas.tests.io.formats.test_to_html import justify # Removed unused import
from rich.console import Console

from src.web.multi_model_utils import get_or_create_model, idle_cleaner, stop_model, list_active_models, stop_all_models
from src.models.vllm_loader import VLLMServer # Import VLLMServer for type hinting and error handling
from src.utils.log_config import logger

app = Flask(__name__)
console = Console()

requests_time_out = 1200

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
                forward_response = requests.post(forward_url, json=data, stream=True, timeout=requests_time_out) # Added timeout
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
                forward_response = requests.post(forward_url, json=data, timeout=requests_time_out) # Added timeout
                forward_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                # Forward the exact response from the vLLM server
                return forward_response.content, forward_response.status_code, forward_response.headers.items()

        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error to VLLM server {model_name} at {forward_url}: {conn_err}")
            # Trigger the restart mechanism
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
            # Forward the request to the vLLM server's embedding endpoint
            forward_response = requests.post(forward_url, json=data, timeout=120) # Timeout for embedding requests
            forward_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Forward the exact response from the vLLM server
            # Note: vLLM embedding response content-type is typically application/json
            return forward_response.content, forward_response.status_code, forward_response.headers.items()

        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error to VLLM server {model_name} (embeddings) at {forward_url}: {conn_err}")
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


def run():
    """CLI entrypoint to run the Flask App."""
    parser = argparse.ArgumentParser(description="Run text-generation model with specified parameters.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the Flask app.')
    parser.add_argument('--port', type=int, default=5824, help='Port for the Flask app.')
    args = parser.parse_args()

    # Start usage-monitor in a thread (daemon = True)
    threading.Thread(target=idle_cleaner, daemon=True).start()

    app.run(host=args.host, port=args.port)
