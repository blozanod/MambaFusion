#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@crc_a10           # Specify queue
#$ -N MambaTraining
#$ -l gpu_card=4
#$ -cwd

# Usage: qsub [-N <job_name>] main/mamba_job.sh <config.yml>
#
# The config path may be relative to the submission directory or absolute:
# it is resolved with realpath before this script changes directory, so the
# `cd` below cannot break it. Override -N per run so concurrent jobs are
# distinguishable in qstat.

set -uo pipefail

REPO=/groups/rls/blozanod/MambaFusion

if [ -z "${1:-}" ]; then
    echo "Error: no config file specified."
    echo "Usage: qsub [-N <job_name>] main/mamba_job.sh <config.yml>"
    exit 1
fi

CONFIG="$(realpath "$1")"
if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $1 (resolved to $CONFIG)"
    exit 1
fi

RUN_NAME="$(awk '/^name:/ {print $2; exit}' "$CONFIG")"

echo "======================================================================"
echo "  Config    : $CONFIG"
echo "  Run name  : ${RUN_NAME:-<unset>}"
echo "  Host      : $(hostname)"
echo "  Job ID    : ${JOB_ID:-<none>}"
echo "  Commit    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Started   : $(date)"
echo "======================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "======================================================================"

conda activate MambaTraining
cd "$REPO/main"

torchrun --nproc_per_node=4 train.py -opt "$CONFIG" --launcher pytorch
STATUS=$?

echo "======================================================================"
echo "  train.py exit status : $STATUS"
echo "  Finished             : $(date)"
echo "======================================================================"

# Post-training analysis: resolves the run's experiment folder from the
# config's `name:` field, finds its newest log, and writes the logfile
# dashboard/summary + checkpoint progress visualization under
# analysis/outputs/<name>/. See analysis/run_analysis.py. Runs even when
# training failed, so a crashed run still leaves a partial dashboard.
python "$REPO/analysis/run_analysis.py" --config "$CONFIG"

# Propagate the training status so a failed run is not reported as success.
exit $STATUS
