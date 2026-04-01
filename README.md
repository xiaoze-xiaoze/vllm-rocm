# AMD AIMAX+ 395 本地部署 vLLM 服务 —— 手动安装与调优指南

本文面向 AMD AIMAX+ 395 环境，记录在 Linux（以 Ubuntu 24.04 为例）上使用 ROCm 手动搭建 vLLM 推理服务的步骤。文中所有涉及代码与命令的内容保持原样，仅对叙述与目录结构做整理，以便复用与排查问题。

参考文档与相关链接：
- ROCm 官方文档： [ROCm 官方文档](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- vLLM 官方文档： [vllm 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/?query=rocm)
- vLLM 源码仓库： [vllm-project/vllm](https://github.com/vllm-project/vllm)
- flash-attention 源码仓库： [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- ROCm manylinux 包索引： [repo.radeon.com/rocm/manylinux](https://repo.radeon.com/rocm/manylinux/)
- ModelScope： [ModelScope](https://modelscope.cn/)
- Hugging Face Hub： [Hugging Face](https://huggingface.co/)

## 一、硬件与系统

建议在部署前先固化基础信息，便于后续复现与定位问题：
- 硬件：AMD AIMAX+ 395
- 操作系统：Ubuntu 24.04（其他发行版请参考 ROCm 官方文档）
- ROCm 版本：7.2
- Python 版本：3.12

## 二、系统级依赖与 ROCm 环境

### 2.1 安装 ROCm 驱动与运行时（ROCm 7.2）
``` bash
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME
sudo apt install rocm
```

### 2.2 验证 ROCm 安装状态
``` bash
rocminfo          # 查看 ROCm 信息
rocm-smi          # 查看 GPU 状态（类似 nvidia-smi）
hipcc --version   # 查看 HIP 编译器版本
```

## 三、Python 环境构建（以 micromamba 为例）

本节使用 micromamba 管理虚拟环境；也可以使用 conda/miniconda 等工具，原则是确保 Python 版本与 ROCm 生态匹配。

相关链接：
- micromamba 文档： [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
- conda 文档： [conda](https://docs.conda.io/projects/conda/en/latest/)
- Miniconda 文档： [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- conda-forge： [conda-forge](https://conda-forge.org/)
- conda 包检索（Anaconda.org）： [Anaconda.org](https://anaconda.org/)
- conda search 命令参考： [conda search](https://docs.conda.io/projects/conda/en/latest/commands/search.html)

### 3.1 创建并激活虚拟环境
``` bash
# 此处使用 micromamba 管理虚拟环境，也可以使用 conda 等其他工具
micromamba create -n vllm-server python=3.12 -c conda-forge -y
micromamba activate vllm-server
```

## 四、包依赖安装（基于对应 ROCm 版本）

### 4.1 安装基础构建工具
``` bash
pip install ninja cmake wheel pybind11
```

### 4.2 安装 ROCm 对应版本的 torch/triton 等组件
从 [https://repo.radeon.com/rocm/manylinux/](https://repo.radeon.com/rocm/manylinux/) 选择与 ROCm 版本、Python 版本匹配的 `triton` `torch` `torchvision` `torchaudio`：
``` bash
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
            https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

## 五、vLLM 源码安装 + 手动管理 vLLM 依赖

本节的目标是以 ROCm 为目标平台完成 vLLM 本地编译，并尽量避免因上游依赖锁定或 CUDA 相关依赖引入导致的编译失败。

### 5.1 本地编译 flash-attention（ROCm）
相关链接：
- flash-attention 源码仓库： [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
``` bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"    # 开启 ROCm 支持
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
export GPU_ARCHS="gfx1151"    # AMD AIMAX+ 395 的 GPU 架构是 gfx1511
python setup.py install
cd ..  
```

### 5.2 安装相关依赖
``` bash
micromamba install setuptools importlib_metadata packaging wheel -c conda-forge -y
micromamba install -y numba scipy numpy
pip install huggingface-hub[cli]
micromamba install modelscope -c conda-forge -y
micromamba install einops pillow transformers -c conda-forge -y
micromamba install -c conda-forge blake3 -y
```

### 5.3 获取 vLLM 源码
``` bash
git clone https://github.com/vllm-project/vllm.git
```

由于 vllm 官方把依赖版本写的很死，容易调用依赖 cuda 的库导致编译失败，所以我们修改部分源码，手动管理依赖

找到 `setup.py` 文件的最后这一段
``` python
setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets", "scipy"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.2.2"],
        "runai": ["runai-model-streamer[s3,gcs] >= 0.15.3"],
        "audio": [
            "librosa",
            "scipy",
            "soundfile",
            "mistral_common[audio]",
        ],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
        "flashinfer": [],  # Kept for backwards compatibility
        # Optional deps for AMD FP4 quantization support
        "petit-kernel": ["petit-kernel"],
        # Optional deps for Helion kernel development
        "helion": ["helion"],
        # Optional deps for OpenTelemetry tracing
        "otel": [
            "opentelemetry-sdk>=1.26.0",
            "opentelemetry-api>=1.26.0",
            "opentelemetry-exporter-otlp>=1.26.0",
            "opentelemetry-semantic-conventions-ai>=0.4.1",
        ],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
```
修改 `setup` 的 `install_requires` 和 `extras_require` 参数
``` python
setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=[],
    extras_require={},
    # install_requires=get_requirements(),
    # extras_require={
    #     "bench": ["pandas", "matplotlib", "seaborn", "datasets", "scipy"],
    #     "tensorizer": ["tensorizer==2.10.1"],
    #     "fastsafetensors": ["fastsafetensors >= 0.1.10"],
    #     "runai": ["runai-model-streamer[s3,gcs] >= 0.15.3"],
    #     "audio": [
    #         "librosa",
    #         "scipy",
    #         "soundfile",
    #         "mistral_common[audio]",
    #     ],  # Required for audio processing
    #     "video": [],  # Kept for backwards compatibility
    #     "flashinfer": [],  # Kept for backwards compatibility
    #     # Optional deps for AMD FP4 quantization support
    #     "petit-kernel": ["petit-kernel"],
    #     # Optional deps for Helion kernel development
    #     "helion": ["helion"],
    #     # Optional deps for OpenTelemetry tracing
    #     "otel": [
    #         "opentelemetry-sdk>=1.26.0",
    #         "opentelemetry-api>=1.26.0",
    #         "opentelemetry-exporter-otlp>=1.26.0",
    #         "opentelemetry-semantic-conventions-ai>=0.4.1",
    #     ],
    # },
    cmdclass=cmdclass,
    package_data=package_data,
)
```

找到 `pyproject.toml` 里的 `[build-system].requires`
``` toml
requires = [
    "cmake>=3.26.1",
    "ninja",
    "packaging>=24.2",
    "setuptools>=77.0.3,<81.0.0",
    "setuptools-scm>=8.0",
    "torch == 2.10.0",
    "wheel",
    "jinja2",
    "grpcio-tools==1.78.0",
]
```
注释掉 `torch == 2.10.0` 这一行
``` toml
requires = [
    "cmake>=3.26.1",
    "ninja",
    "packaging>=24.2",
    "setuptools>=77.0.3,<81.0.0",
    "setuptools-scm>=8.0",
    # "torch == 2.10.0",
    "wheel",
    "jinja2",
    "grpcio-tools==1.78.0",
]
```

手动管理 vllm 需要的依赖
``` bash
# 基础层：运行时 + 网络 + Web
# Base Layer: Runtime + Networking + Web Stack
# Base Layer: Runtime + Networking + Web Stack
micromamba install -y uvloop aiohttp pyzmq fastapi prometheus_client prometheus-fastapi-instrumentator

# AI 层：模型 + 推理 + 约束
# AI Layer: Models + Inference + Structured Output Constraints
micromamba install -y transformers gguf openai openai-harmony llguidance mistral-common lm-format-enforcer

# 工具层：数据 + 缓存 + 工具
# Tooling Layer: Data Processing + Caching + System Utilities
micromamba install -y pydantic msgspec cbor2 ijson partial-json-parser psutil py-cpuinfo diskcache cachetools cloudpickle pillow pybase64 regex grpcio-tools

# 别用 micromamba 安装 xgrammar，会删除 rocm 版本的 pytorch
# Dont use mamba to install xgrammar, or will delete rocm version of pytorch
pip install xgrammar

# 别用 micromamba 安装，用 pip 安装
# Dont exists in mamba, install from pip
pip install setuptools_scm model-hosting-container-standards amdsmi
```

### 5.4 本地编译 vLLM（ROCm）
``` bash
cd vllm
export PYTORCH_ROCM_ARCH="gfx1151"
export VLLM_TARGET_DEVICE=rocm
python3 setup.py install
cd ..
```

## 六、模型下载与服务启动

### 6.1 下载模型
```bash
python scripts/download.py -m Qwen/Qwen3-VL-8B-Instruct --source modelscope    # --source 可选 huggingface
```

### 6.2 启动服务
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
vllm serve models/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b-instruct --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.8 --port 8000 --api-key "sk-123456" --trust-remote-code
```
