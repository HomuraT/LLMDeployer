import openai
from sglang.test.test_utils import is_in_ci
import sys
import os

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process


server_process, port = launch_server_cmd(
    f"python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0",
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")

print('### prepare client')
# client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
#
# print('### send message')
# response = client.chat.completions.create(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     messages=[
#         {"role": "user", "content": "List 3 countries and their capitals."},
#     ],
#     temperature=0,
#     max_tokens=64,
# )
# print('### response')
# print_highlight(f"Response: {response}")
# print('### shutdown server')