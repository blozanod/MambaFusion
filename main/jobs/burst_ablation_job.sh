#!/bin/bash

#$ -M blozanod@nd.edu
#$ -m abe
#$ -pe smp 32
#$ -q gpu@@crc_a10
#$ -N BurstAblation
#$ -l gpu_card=4
#$ -cwd

# Usage:
#   qsub burst_ablation_job.sh <path_to_checkpoint>
#
# Example:
#   qsub burst_ablation_job.sh ../experiments/STHAT_GW/models/net_g_35000.pth

if [ -z "$1" ]; then
    echo "Error: No checkpoint path specified."
    echo "Usage: qsub burst_ablation_job.sh <path_to_checkpoint>"
    exit 1
fi

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion

torchrun --nproc_per_node=4 analysis/burst_ablation.py \
    --model_path "$1" \
    --config main/config.yml \
    --data_root /groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch \
    --seed 42 \
    --crop_border 40 \
    --num_frames 5 \
    --log_dir /groups/rls/blozanod/MambaFusion/analysis/ablation_logs
