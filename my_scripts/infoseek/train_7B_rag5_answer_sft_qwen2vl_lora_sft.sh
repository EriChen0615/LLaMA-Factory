#!/bin/bash
#SBATCH -J infoseek_train_rag5_answer_sft_qwen2vl_7B_lora_sft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC"

which python

llamafactory-cli train my_configs/infoseek/rag5_answer_sft_qwen2vl-7B_lora_sft.yaml