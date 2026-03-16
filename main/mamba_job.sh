#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@crc_a10           # Specify queue
#$ -l gpu_card=4
#$ -N MambaFusion_Train       # Specify job name
#$ -cwd

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion/main

torchrun --nproc_per_node=4  train.py -opt config.yml --launcher pytorch
