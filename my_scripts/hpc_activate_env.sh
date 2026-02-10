#!/bin/bash
module purge
module load slurm
module load rhel8/default-amp
# module load cuda/11.4
module load cuda/12.1
module load gcc/9

export HF_HOME='../../../HF_HOME'
# export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
conda activate /rds/project/rds-hirYTW1FQIw/shared_space/envs/LlamaFactory
# conda activate /rds/project/wjb31/rds-wjb31-nmt2020/shared_drive/to0