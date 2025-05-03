import os

def huggingface_use_domestic_endpoint()->None:
    '''
    **需要在代码第一行运行，否则会无效**

    export HF_ENDPOINT=https://hf-mirror.com
    :return:
    '''
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def set_python_path():
    os.environ["PATH"] = f"/home/renlin/miniconda3/envs/LLMDeployer/bin:{os.environ['PATH']}"