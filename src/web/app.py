import argparse
import threading
import time

from flask import Flask, request, jsonify
from rich.console import Console
from rich.panel import Panel
from datetime import datetime

from src.models.model_worker import monitor_model_usage
from src.models.multi_model_manage import ensure_model_process

app = Flask(__name__)
console = Console()

@app.route('/generate', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    """Endpoint for generating text from a loaded model."""
    data = request.json
    model_name = data.get('model')
    if not model_name:
        return jsonify({'error': 'Model parameter is required.'}), 400

    # Spawn or retrieve an existing model worker process
    ensure_model_process(model_name)

    # Extract parameters from request body
    messages = data.get('messages', [])
    max_new_tokens = data.get('max_new_tokens', 512)
    do_sample = data.get('do_sample', False)
    options = data.get('options', {})
    temperature = options.get('temperature', 0.01)
    top_k = data.get('top_k', 100)
    top_p = data.get('top_p', 0.95)
    response_format = data.get('response_format', None)

    model_info = models[model_name]
    model_info['request_queue'].put((
        messages,
        max_new_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
        response_format
    ))

    response = model_info['response_queue'].get()
    # Escape brackets in user/assistant content for safe rendering
    if 'error' in response:
        # If an error occurred in the worker
        return jsonify(response), 500

    user_content = messages[0]['content'].replace('[', '\\[') if messages else ""
    assistant_content = response['content'].replace('[', '\\[')

    # Logging to console with nice formatting
    content = (
        f"[bold dark_blue]{model_name}[/bold dark_blue]\n"
        f"{'-' * 50}\n"
        f"[bold dark_red]Parameters[/bold dark_red]\n"
        f"[bold blue]max_new_tokens[/bold blue]: [black]{max_new_tokens}[/black]\n"
        f"[bold blue]do_sample[/bold blue]: [black]{do_sample}[/black]\n"
        f"[bold blue]temperature[/bold blue]: [black]{temperature}[/black]\n"
        f"[bold blue]top_k[/bold blue]: [black]{top_k}[/black]\n"
        f"[bold blue]top_p[/bold blue]: [black]{top_p}[/black]\n"
        f"[bold blue]response_format[/bold blue]: [black]{response_format}[/black]\n"
        f"{'=' * 50}\n"
        f"[bold dark_red]User[/bold dark_red]:\n"
        f"[black]\"{user_content}\"[/black]\n"
        f"{'-' * 50}\n"
        f"[bold dark_red]Assistant[/bold dark_red]:\n"
        f"[black]\"{assistant_content}\"[/black]\n"
        f"{'=' * 50}\n"
        f"[bold green]Time[/bold green]: [black]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/black]"
    )
    console.print(
        Panel(
            content,
            title="[bold dark_blue]LLM Chat[/bold dark_blue]",
            border_style="blue",
            style="white",
            subtitle="[italic grey39]Chat Log[/italic grey39]",
            subtitle_align="right"
        )
    )

    # Update last request time for this model
    model_info['last_request_time'] = time.time()

    return jsonify({'message': {'content': assistant_content}})

def main():
    """CLI entrypoint to run the Flask App."""
    parser = argparse.ArgumentParser(description="Run text-generation model with specified parameters.")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the Flask app.')
    parser.add_argument('--port', type=int, default=5001, help='Port for the Flask app.')
    args = parser.parse_args()

    # Start usage-monitor in a thread (daemon = True)
    threading.Thread(target=monitor_model_usage, daemon=True).start()

    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()