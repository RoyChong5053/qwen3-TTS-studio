#!/bin/bash
cd "$(dirname "$0")"

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7860

# Flash attention 不可用时会自动降级，设置以下变量优化性能
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

# 启动应用
python qwen_tts_ui.py
