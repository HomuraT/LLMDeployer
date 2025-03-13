# install
[SGL install document](https://docs.sglang.ai/start/install.html)
```
pip install -r requirement.txt
uv pip install "sglang[all]>=0.4.3.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
conda install ninja
```

# Error
当遇到`../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`
可运行：
```shell
python -m pip uninstall torch torchvision torchaudio
python -m pip install --pre torch torchvision torchaudio 
```