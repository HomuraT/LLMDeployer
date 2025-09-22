# LLMDeployer: 开源LLM便捷部署框架

LLMDeployer 是一个旨在简化开源大型语言模型（LLM）部署和管理的框架。它解决了本地部署 LLM 时常见的资源管理、多模型并发以及与现有 API 生态（如 OpenAI Function Calling）集成的问题。

**主要特性:**

*   **智能 GPU 选择:** 自动检测并利用空闲的 GPU 资源进行模型加载，优化资源利用率。
*   **多模型并发通讯:** 通过端口转发机制，允许同时与部署的多个 LLM 进行交互。
*   **资源自动释放:** 当 LLM 在设定时间内无交互时，自动关闭并释放所占用的 GPU 资源，降低闲置成本。
*   **兼容 OpenAI Function Calling:** 通过输入输出适配层，无缝对接 OpenAI 的 Function Calling 接口，方便集成到现有应用。

# 安装 (Installation)
[SGLang 安装文档](https://docs.sglang.ai/start/install.html)
```shell
conda create -n LLMDeployer python=3.11 -y
conda activate LLMDeployer

# 安装依赖 (Install dependencies)
pip install -r requirement.txt
# 推荐使用清华源加速 (Using Tsinghua mirror is recommended for acceleration)
uv pip install "sglang[all]>=0.4.3.post4" -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install ninja -y
```

# 常见错误 (Common Errors)
当遇到 `../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12` 错误时，通常是 PyTorch 版本与 CUDA Driver 不兼容导致。
可尝试运行以下命令重新安装 PyTorch：
```shell
python -m pip uninstall torch torchvision torchaudio -y
# 安装适用于您 CUDA 版本的 PyTorch (Install PyTorch compatible with your CUDA version)
# 例如，如果您使用 CUDA 12.1 (For example, if you are using CUDA 12.1):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 您可以访问 PyTorch 官网查找适合您环境的安装命令 (You can visit the official PyTorch website to find the appropriate installation command for your environment)
```


# 如何运行 (How to Run)
```shell
conda activate LLMDeployer
python run_api.py
```
服务启动后，你可以通过指定的 API 端点与部署的 LLM 进行交互。

# 设置魔搭社区缓存地址
```shell
vi ~/.bashrc
export MODELSCOPE_CACHE=/Users/xx/tools/cache/modelscope


export HF_DATASETS_CACHE=/Users/xx/tools/cache/huggingface
source ~/.bashrc
```
# 待办事项 (Todo)
*   [ ] 支持嵌入模型 (Support Embedding Models)
*   [ ] 支持多模态模型 (Support Multimodal Models)