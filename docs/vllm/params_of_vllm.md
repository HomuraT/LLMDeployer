`vllm==0.7.2`
# vllm.LLM
```python
def __init__(
    self,
    model: str,
    tokenizer: Optional[str] = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: Optional[bool] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    disable_async_output_proc: bool = False,
    hf_overrides: Optional[HfOverrides] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    # After positional args are removed, move this right below `model`
    task: TaskOption = "auto",
    override_pooler_config: Optional[PoolerConfig] = None,
    compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
    **kwargs,
) 
```
- model  
  英文描述: The name or path of a HuggingFace Transformers model.  
  中文翻译: 模型名称或HuggingFace Transformers 模型在本地或在线的路径。  
  用法: 用于指定要加载的模型来源, 以便进行推理或微调。

- tokenizer  
  英文描述: The name or path of a HuggingFace Transformers tokenizer.  
  中文翻译: 分词器名称或HuggingFace Transformers 分词器在本地或在线的路径。  
  用法: 用于解析输入文本, 将其分割成可供模型理解的词元。

- tokenizer_mode  
  英文描述: The tokenizer mode. “auto” will use the fast tokenizer if available, and “slow” will always use the slow tokenizer.  
  中文翻译: 分词器模式, "auto" 会在可用时使用快速分词器, “slow” 则始终使用慢速分词器。  
  用法: 根据实际需求在性能和兼容性之间进行平衡选择。

- skip_tokenizer_init  
  英文描述: If true, skip initialization of tokenizer and detokenizer. Expect valid prompt_token_ids and None for prompt from the input.  
  中文翻译: 如果为 true, 跳过初始化分词器和反分词器, 期望从输入直接提供有效的 prompt_token_ids 而无需 prompt 文本。  
  用法: 在高级场景下可用来节省初始化时间, 前提是有可用的词元 ID。

- trust_remote_code  
  英文描述: Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.  
  中文翻译: 当从远程(如 HuggingFace)下载模型与分词器时, 信任并执行远程代码。  
  用法: 在安全环境中启用, 允许自动执行相关的远程代码以便下载和初始化模型。

- allowed_local_media_path  
  英文描述: Allowing API requests to read local images or videos from directories specified by the server file system. This is a security risk. Should only be enabled in trusted environments.  
  中文翻译: 允许从服务器文件系统指定的目录中读取本地图片或视频, 存在安全风险, 仅在可信环境中启用。  
  用法: 开启后可实现对本地多媒体文件的访问, 但需注意安全风险。

- tensor_parallel_size  
  英文描述: The number of GPUs to use for distributed execution with tensor parallelism.  
  中文翻译: 使用张量并行时所使用的 GPU 数量。  
  用法: 在多 GPU 环境中分配不同的 GPU 共同完成推理或训练, 提高吞吐量。

- dtype  
  英文描述: The data type for the model weights and activations. Currently, we support float32, float16, and bfloat16. If auto, we use the torch_dtype attribute specified in the model config file. However, if the torch_dtype in the config is float32, we will use float16 instead.  
  中文翻译: 模型权重和激活的精度类型。目前支持 float32、float16 和 bfloat16。如果是 auto, 将使用模型配置文件中的 torch_dtype 属性；但如果配置文件里是 float32, 则会被覆盖为 float16。  
  用法: 根据硬件支持与内存限制选择合适的精度, 以在速度和精度之间进行平衡。

- quantization  
  英文描述: The method used to quantize the model weights. Currently, we support “awq”, “gptq”, and “fp8” (experimental). If None, we first check the quantization_config attribute in the model config file. If that is None, we assume the model weights are not quantized and use dtype to determine the data type of the weights.  
  中文翻译: 模型量化的方法。目前支持 “awq”、 “gptq” 和(实验性的) “fp8”。如果为 None, 则会先检查模型配置文件中的 quantization_config 属性；若依然未发现量化配置, 将假设模型未进行量化并使用 dtype 指定的类型。  
  用法: 通过量化方法可有效地降低显存占用, 并在一定程度上提高推理速度。

- revision  
  英文描述: The specific model version to use. It can be a branch name, a tag name, or a commit id.  
  中文翻译: 要使用的特定模型版本, 可以是分支名称、标签或提交 ID。  
  用法: 在模型仓库中精确指定需要的版本, 以保证可重复的结果。

- tokenizer_revision  
  英文描述: The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id.  
  中文翻译: 要使用的特定分词器版本, 可以是分支名称、标签或提交 ID。  
  用法: 在分词器仓库中精确指定需要的版本, 以保证与模型版本一致。

- seed  
  英文描述: The seed to initialize the random number generator for sampling.  
  中文翻译: 用于初始化随机数生成器的种子。  
  用法: 在采样时保证结果可复现, 使调试和对比实验更方便。

- gpu_memory_utilization  
  英文描述: The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model’s throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.  
  中文翻译: 在 0 到 1 之间指定分配给模型权重、激活值和 KV cache 的 GPU 显存比例。比例越大, KV cache 就越大, 可提高吞吐量, 但过大可能导致内存不足报错。  
  用法: 根据实际 GPU 显存大小和负载, 合理设置此参数以平衡性能和稳定性。

- swap_space  
  英文描述: The size (GiB) of CPU memory per GPU to use as swap space. This can be used for temporarily storing the states of the requests when their best_of sampling parameters are larger than 1. If all requests will have best_of=1, you can safely set this to 0. Noting that best_of is only supported in V0. Otherwise, too small values may cause out-of-memory (OOM) errors.  
  中文翻译: 每块 GPU 可使用的 CPU 内存大小(以 GiB 为单位), 用作交换空间。当在请求中使用 best_of>1 的参数时, 可以将请求的部分状态暂存在这块 CPU 内存里。如果所有请求均为 best_of=1, 可以将其设为 0。最佳采样模式仅在 V0 中支持, 若设置过小可能导致内存不足。  
  用法: 在多样本对比采样场景下调大此值, 避免 GPU 内存不足, 提高安全性与稳定性。

- cpu_offload_gb  
  英文描述: The size (GiB) of CPU memory to use for offloading the model weights. This virtually increases the GPU memory space you can use to hold the model weights, at the cost of CPU-GPU data transfer for every forward pass.  
  中文翻译: 用于将模型权重卸载到 CPU 的内存大小(以 GiB 为单位)。这可以在一定程度上扩展可用于存放模型权重的 GPU 内存, 但需要在每次前向传递时进行 CPU-GPU 间的数据传输, 影响吞吐。  
  用法: 在 GPU 显存不足时, 可通过增加 CPU 内存来延缓 OOM 问题, 但可能会降低推理速度。

- enforce_eager  
  英文描述: Whether to enforce eager execution. If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph and eager execution in hybrid.  
  中文翻译: 是否强制启用 eager 执行。如果为 True, 禁用 CUDA graph 并始终以 eager 模式执行模型; 若为 False, 则采用 CUDA graph 与 eager 的混合模式。  
  用法: 在调试或某些特定场景需要禁用 CUDA graph 时可启用, 以简化执行模式。

- max_seq_len_to_capture  
  英文描述: Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode.  
  中文翻译: CUDA graph 所能覆盖的最大序列长度。当序列的上下文长度超过此值时, 会回退到 eager 模式。对于编码器-解码器模型, 如果编码器输入序列长度超过此值, 同样会回退到 eager 模式。  
  用法: 优化较短序列的执行效率, 同时确保超长序列在无法满足 CUDA graph 条件时依然可被处理。

- disable_custom_all_reduce  
  英文描述: See ParallelConfig  
  中文翻译: 参见 ParallelConfig 配置, 用以禁用自定义的 all_reduce 操作。  
  用法: 在分布式训练或推理中的特殊场景可能需要禁用自定义 all_reduce, 以保障兼容性或执行稳定性。

- disable_async_output_proc  
  英文描述: Disable async output processing. This may result in lower performance.  
  中文翻译: 禁用异步输出处理, 可能导致性能下降。  
  用法: 在需要完全同步处理输入输出的场景下可启用, 保证顺序性但牺牲并发效率。

- hf_overrides  
  英文描述: If a dictionary, contains arguments to be forwarded to the HuggingFace config. If a callable, it is called to update the HuggingFace config.  
  中文翻译: 如果是字典, 则包含要传递给 HuggingFace 配置的参数。如果是可调用对象, 则会调用它来更新 HuggingFace 配置。  
  用法: 在高级场景下灵活调整 HuggingFace 配置, 例如更改一些默认设置或添加自定义参数等。

- compilation_config  
  英文描述: Either an integer or a dictionary. If it is an integer, it is used as the level of compilation optimization. If it is a dictionary, it can specify the full compilation configuration.  
  中文翻译: 可以是整数或字典。如果是整数, 表示编译优化级别；如果是字典, 可指定完整的编译配置信息。  
  用法: 用于控制编译优化程度, 提高推理或训练性能, 也能根据需求进行自定义配置。

- **kwargs (EngineArgs)  
  英文描述: Arguments for EngineArgs. (See Engine Arguments)  
  中文翻译: 将额外的参数传递给 EngineArgs。(详见 Engine Arguments)  
  用法: 在需要额外设置引擎参数时使用, 例如自定义日志级别、部署选项等。

# EngineArgs
https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args