#!/bin/bash
#SBATCH -J train_norag_answer_qwen2vl-7B_full_sft
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
which python

FORCE_TORCHRUN=1 llamafactory-cli train my_configs/norag_answer_qwen2vl-7B_full_sft.yaml