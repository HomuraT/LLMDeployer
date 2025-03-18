import re
import subprocess

def find_available_gpu(model_name=None, min_memory_mb=24000):
    """
    Find an available GPU based on free memory, optionally adjusting min_memory_mb if model is small/large.
    """
    if model_name:
        # Regex to find a pattern like "7B", "13B", "30B", etc., from the model name
        pattern = r'(\d+(?:\.\d+)?)B'
        b_num = re.findall(pattern, model_name)
        if b_num:
            b_val = float(b_num[0])
            if b_val < 7:
                min_memory_mb = 10000  # smaller model => smaller memory requirement
            if b_val < 1:
                min_memory_mb = 3000   # very small model => even smaller

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("Failed to execute nvidia-smi:", result.stderr)
            return None

        free_memory = [int(x) for x in result.stdout.strip().split('\n')]
        max_memory = -1
        selected_gpus = []

        for i, mem in enumerate(free_memory):
            if mem >= min_memory_mb and mem >= max_memory:
                max_memory = mem
                selected_gpus.append(i)

        if selected_gpus:
            print(f"GPU {selected_gpus} has enough memory: {max_memory} MB free.")
            return selected_gpus
        else:
            print(f"No GPU found with minimum memory of {min_memory_mb} MB free.")
            return None

    except Exception as e:
        print("Error finding available GPU:", e)
        return None