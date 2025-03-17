# config path of vllm models
import os

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '../../')
VLLM_MODEL_CONFIG_BASE_PATH = os.path.join(PROJECT_PATH, 'resources', 'vllm_models')