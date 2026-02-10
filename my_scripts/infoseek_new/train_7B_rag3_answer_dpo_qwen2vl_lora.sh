#!/bin/bash
#SBATCH -J infoseek_new_train_7B_rag1_answer_dpo_qwen2vl_lora
#SBATCH -A BYRNE-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-3 # Adjust this based on the number of experiments
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
export WANDB_RUN_GROUP="HPC"
# BETAS=(0.1 0.5 0.7 1.0 1.2 1.5)
BETAS=(0.1 0.5 1.0 2.0)
# BETAS=(0.1)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

which python

llamafactory-cli train my_configs/infoseek_new/rag3_answer_dpo_qwen2vl-7B_lora_beta=${BETA}.yaml
