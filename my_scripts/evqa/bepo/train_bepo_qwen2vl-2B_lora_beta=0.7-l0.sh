#!/bin/bash
#SBATCH -A BYRNE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=5:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC-PPL"

which python

# llamafactory-cli train my_configs/evqa/bepo/bepo_qwen2vl-2B_lora_beta=0.7.yaml
llamafactory-cli train my_configs/evqa/bepo/bepo_qwen2vl-2B_lora_beta=0.7-l0.yaml
