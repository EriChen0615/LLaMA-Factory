#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC"

which python

llamafactory-cli train my_configs/slidevqa/beft/beft[K=4*]-7B-prior=dpp-fusedgt_qwen2vl-7B_lora_r64_bs8_size=0.yaml
