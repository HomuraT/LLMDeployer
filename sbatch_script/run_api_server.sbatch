#!/bin/bash

#SBATCH --job-name=vllm_api_server  # Job name
#SBATCH --partition=qigl            # Your specified partition
#SBATCH --nodes=1                   # Requesting 1 node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=24          # CPUs per task
#SBATCH --gres=gpu:2                # Requesting 1 GPU
#SBATCH --output=slurm_output/%j_vllm_api_server.out  # Standard output file
#SBATCH --error=slurm_output/%j_vllm_api_server.err   # Standard error file
#SBATCH --mail-type=FAIL            # Optional: Notify on FAIL
#SBATCH --mail-user=xxxx@seu.edu.cn   # Optional: Your email (change xxxx)

# Load necessary modules
module load anaconda3-2024.10-1
module load cuda-12.4

# Navigate to a working directory.
# This could be where your script is, or a general project directory.
# For this example, let's assume the Python script is in your home's script folder or similar.
# Adjust PYTHON_SCRIPT_PATH and the cd command as needed.
# If run_api.py is in the same directory as this slurm script,
# you might not even need to cd, or cd to its location.

# Example: if python script is in /seu_share/home/qiguilin/230248999/my_scripts/
# cd /seu_share/home/qiguilin/230248999/my_scripts/
# PYTHON_SCRIPT_NAME="run_api.py"

# Simpler: Assume the python script is in the directory where you submit the slurm job,
# or that its path is correctly specified below.
# For this example, let's assume the script is placed where the Qwen model is,
# which is a bit unusual but matches one interpretation of your structure.
# Better to keep scripts separate from model data if possible.

# Let's assume the python script is in:
# /seu_share/home/qiguilin/230248999/scripts/run_api.py
# And you run sbatch from /seu_share/home/qiguilin/230248999/scripts/
# Then the output directory slurm_output will be created in /seu_share/home/qiguilin/230248999/scripts/

PYTHON_SCRIPT_NAME="run_api.py" # Ensure this script is in the correct WORK_DIR for your setup

# Activate Conda environment (LLMDeployer for vLLM)
source activate LLMDeployer

# Run the Python script for vLLM API server
# The -u flag ensures unbuffered Python output
echo "Starting Python script: $PYTHON_SCRIPT_NAME from $(pwd)"
python -u $PYTHON_SCRIPT_NAME # Added -u for unbuffered output, common for servers

echo "Job Finished (or server started and running in background if configured)" 