# install
[SGL install document](https://docs.sglang.ai/start/install.html)
```
pip install -r requirement.txt
uv pip install "sglang[all]>=0.4.3.post4" -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install ninja
```

# Error
当遇到`../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`
可运行：
```shell
python -m pip uninstall torch torchvision torchaudio
python -m pip install --pre torch torchvision torchaudio 
```