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

PYTHON_SCRIPT_NAME="run_api.py" # Ensure this script is in the correct WORK_DIR for your setup

# Activate Conda environment (LLMDeployer for vLLM)
source activate LLMDeployer

# Run the Python script for vLLM API server
# The -u flag ensures unbuffered Python output
echo "Starting Python script: $PYTHON_SCRIPT_NAME from $(pwd)"
python $PYTHON_SCRIPT_NAME # Added -u for unbuffered output, common for servers

echo "Job Finished (or server started and running in background if configured)" 