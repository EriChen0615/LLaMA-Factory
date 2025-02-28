#!/bin/bash

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0  # Adjust based on available GPUs
export PYTHONPATH="$PYTHONPATH:$(pwd)"
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Prepare the dataset first
# python data/jinghong_chen/prepare_gsm8k.py

# Run the training with FSDP
# llamafactory-cli train my_configs/gsm8k/sft_0.5B.yaml
llamafactory-cli train my_configs/gsm8k/sft_3B.yaml

# Optional: Run evaluation after training
# python src/train_bash.py \
#     my_configs/gsm8k/sft.yaml \
#     --eval_dataset gsm8k \
#     --eval_split test \
#     --predict_with_generate true \
#     --max_samples 100 