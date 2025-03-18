import argparse
import threading

import requests
from flask import Flask, request, jsonify
from pandas.tests.io.formats.test_to_html import justify
from rich.console import Console

from src.web.multi_model_utils import get_or_create_model, idle_cleaner

app = Flask(__name__)
console = Console()

@app.route('/generate', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    """Endpoint for proxying text-generation requests to another service."""
    data = request.json
    model_name = data['model']
    llm = get_or_create_model(model_name)
    forward_url = f"http://127.0.0.1:{llm.port}/v1/chat/completions"

    if 'functions' in data and data['functions']:
        # Variable conversion adaptation of functions calling to open AI input format
        data['tools'] = [{'type':'function', 'function': i} for i in data['functions']]
        data['tool_choice'] = data['function_call']

    response = requests.post(forward_url, json=data)

    if 'functions' in data and data['functions']:
        # Adapting the variable conversion of functions calling to the output format of open AI
        response_json = response.json()
        response_json['choices'][0]['message']['function_call'] = response_json['choices'][0]['message']['tool_calls'][0]['function']
        del response_json['choices'][0]['message']['tool_calls']
        response_bytes = jsonify(response_json).get_data()
        return response_bytes, response.status_code, response.headers.items()

    return response.content, response.status_code, response.headers.items()

def run():
    """CLI entrypoint to run the Flask App."""
    parser = argparse.ArgumentParser(description="Run text-generation model with specified parameters.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the Flask app.')
    parser.add_argument('--port', type=int, default=5824, help='Port for the Flask app.')
    args = parser.parse_args()

    # Start usage-monitor in a thread (daemon = True)
    threading.Thread(target=idle_cleaner, daemon=True).start()

    app.run(host=args.host, port=args.port)