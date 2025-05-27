#!/bin/bash

echo -e "\n\nCreating Conda Env"
# conda create -n drdpo_env python=3.11 gcc_linux-64=13.3.0 gxx_linux-64=13.3.0 binutils=2.42 libstdcxx-ng=13.3.0 nccl=2.22.3 cuda-version=12.4 -c conda-forge -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drdpo_env

echo -e "\n\nInstalling Torch"
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo -e "\n\nInstalling Deepspeed"
git clone https://github.com/microsoft/DeepSpeed.git --branch v0.16.4 -c advice.detachedHead=false
cd DeepSpeed && DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 pip install . && cd ..
rm -rf DeepSpeed

echo -e "\n\nInstalling OpenRLHF"
pip install openrlhf[vllm]==0.6.1.post1

echo -e "\n\nInstalling LM Evaluation Harness"
git clone https://github.com/EleutherAI/lm-evaluation-harness && git checkout 19ba1b16fef9fa6354a3e4ef3574bb1d03552922
cd lm-evaluation-harness && pip install -e ".[math]" && cd ..
rm -rf lm-evaluation-harness

echo -e "\n\nDownloading Models and Datasets"
python src/setup.py