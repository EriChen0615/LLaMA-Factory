#!/bin/bash
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

llamafactory-cli train my_configs/slidevqa/beft/beft[K=4*]-7B-prior=mlp_lr1e-6-l1h4_qwen2vl-7B_lora_r64_bs8_size=64-da=subdivide4.yaml

