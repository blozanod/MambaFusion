#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@crc_a10           # Specify queue
#$ -N MambaTraining
#$ -l gpu_card=4
#$ -cwd

if [ -z "$1" ]; then
    echo "Error: No config file specified. Usage: qsub script.sh <config_name>"
    exit 1
fi

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion/main

torchrun --nproc_per_node=4  train.py -opt "$1" --launcher pytorch

# Post-training analysis: runs once the job's GPU allocation is still held,
# resolves the run's experiment folder from the config's `name:` field, and
# writes the logfile dashboard/summary + checkpoint progress visualization
# under analysis/outputs/<name>/. See analysis/run_analysis.py.
python /groups/rls/blozanod/MambaFusion/analysis/run_analysis.py --config "$1"
