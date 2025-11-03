#!/bin/bash
#SBATCH -A BYRNE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC-PPL"

which python

llamafactory-cli train my_configs/evqa/beft/beft[K=2*]-prior=mlp_lr1e-6-l1h4_qwen2vl-2B_lora_r64.yaml