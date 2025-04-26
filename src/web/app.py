import argparse
import logging
import threading

import requests
from flask import Flask, request, Response, jsonify
from pandas.tests.io.formats.test_to_html import justify
from rich.console import Console

from src.web.multi_model_utils import get_or_create_model, idle_cleaner

app = Flask(__name__)
console = Console()

@app.route('/generate', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    data = request.json
    # logging.info('收到消息：'+str(data))
    model_name = data['model']
    llm = get_or_create_model(model_name)
    forward_url = f"http://127.0.0.1:{llm.port}/v1/chat/completions"

    if data.get('stream'):
        forward_response = requests.post(forward_url, json=data, stream=True)

        def event_stream():
            for chunk in forward_response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

        return Response(event_stream(), content_type='application/json')
    else:
        forward_response = requests.post(forward_url, json=data)
        return forward_response.content, forward_response.status_code, forward_response.headers.items()


def run():
    """CLI entrypoint to run the Flask App."""
    parser = argparse.ArgumentParser(description="Run text-generation model with specified parameters.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the Flask app.')
    parser.add_argument('--port', type=int, default=5824, help='Port for the Flask app.')
    args = parser.parse_args()

    # Start usage-monitor in a thread (daemon = True)
    threading.Thread(target=idle_cleaner, daemon=True).start()

    app.run(host=args.host, port=args.port)