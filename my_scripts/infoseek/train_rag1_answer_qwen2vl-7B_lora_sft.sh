#!/bin/bash
#SBATCH -J train_infoseek_rag1_answer_qwen2vl-7B_lora_sft
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
llamafactory-cli train my_configs/infoseek/rag1_answer_qwen2vl-7B_lora_sft.yaml