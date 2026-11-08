#!/bin/bash
set -e
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda create -n qwen_vl python=3.11 -y
conda activate qwen_vl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install qwen-vl-utils
pip install ms-swift[all] -U
pip install transformers accelerate peft
