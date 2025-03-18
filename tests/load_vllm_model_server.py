import time

from src.utils.enviroment_utils import huggingface_use_domestic_endpoint, set_python_path
from src.utils.gpu_utils import find_available_gpu

huggingface_use_domestic_endpoint()
set_python_path()

from src.models.vllm_loader import load_model, VLLMServer
import threading

llm = VLLMServer("meta-llama/Llama-3.1-8B-Instruct", cuda=find_available_gpu())
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
def worker(idx):
    outputs = llm.chat(messages=conversation)
    print(f"Thread {idx}, outputs: {outputs}")

threads = []
num_threads = 1000  # set how many threads you want
start = time.time()
for i in range(num_threads):
    t = threading.Thread(target=worker, args=(i, ))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
elapsed = time.time() - start
print(f"All threads done. Elapsed: {elapsed:.2f}s")