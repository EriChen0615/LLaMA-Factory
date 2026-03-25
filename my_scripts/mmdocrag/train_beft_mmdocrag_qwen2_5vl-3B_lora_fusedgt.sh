#!/bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -p ampere

export WANDB_RUN_GROUP="HPC-BEFT"

which python

llamafactory-cli train my_configs/mmdocrag/beft_rag8-mmdocrag_qwen2_5vl-3B_lora_r64_bs8_size=0_multimodal_fusedgt.yaml
