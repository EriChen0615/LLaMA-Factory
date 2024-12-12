#!/bin/bash
#SBATCH -J okvqa_train_rag5_answer_sft_qwen2vl_lora_sft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC"

which python

llamafactory-cli train my_configs/okvqa/rag5_answer_sft_qwen2vl-2B_lora_sft.yaml