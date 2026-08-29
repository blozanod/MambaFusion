#!/bin/bash

#$ -M blozanod@nd.edu
#$ -m abe
#$ -pe smp 32
#$ -q gpu@@crc_a10
#$ -N L3Ablations
#$ -l gpu_card=4
#$ -hold_jid 1384006
#$ -cwd

# Runs the SyntheticBurst-compatible L5 ablations against the L3 run
# (MF_STHAT_L3_SynBase), back to back on one GPU allocation:
#   1. burst_ablation.py --mode two_pass    (normal vs. all-ref delta)
#   2. burst_ablation.py --mode frame_drop  (N = 1,2,5,9,14 curve)
#
# fusion_attention_mass.py and exposure_drift.py are NOT included: they
# only support RealBSR-format data (hardcoded .pkl + *_x1_*.png loading,
# default num_frames=5) and would crash against a num_frames=14
# SyntheticBurst checkpoint (fixed-size temporal positional embedding /
# 3D relative-position tables baked in at that frame count).
#
# -hold_jid MambaTraining waits for any currently-running/pending job
# named MambaTraining (main/mamba_job.sh's #$ -N) to finish first. If no
# such job is active when this is submitted, it starts immediately.
#
# Usage: qsub l3_ablations_job.sh

# --- Edit both of these together if you want fewer than 4 GPUs ---
NPROC=4
# --- and keep #$ -l gpu_card=N above in sync with NPROC ---

EXPERIMENT_DIR="/groups/rls/blozanod/MambaFusion/experiments/MF_STHAT_L3_SynBase"
CONFIG="/groups/rls/blozanod/MambaFusion/main/configs/MF_STHAT_L3_SynBase.yml"
VAL_ROOT="/groups/rls/blozanod/MambaFusion/dataset/SyntheticBurstVal"

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion

# --- Resolve the checkpoint to analyze ---
# Prefers the final checkpoint (net_g_-1.pth, written at the very end of
# a completed 100k run) over any numbered periodic checkpoint; falls back
# to iter 60000 if the models/ dir has neither (e.g. wrong path).
CKPT=$(python3 - <<EOF
import glob, os, re
models_dir = "$EXPERIMENT_DIR/models"
final = os.path.join(models_dir, "net_g_-1.pth")
if os.path.isfile(final):
    print(final)
else:
    numbered = []
    for p in glob.glob(os.path.join(models_dir, "net_g_*.pth")):
        m = re.fullmatch(r"net_g_(\d+)\.pth", os.path.basename(p))
        if m:
            numbered.append((int(m.group(1)), p))
    if numbered:
        print(max(numbered)[1])
EOF
)

if [ -z "$CKPT" ]; then
    CKPT="$EXPERIMENT_DIR/models/net_g_60000.pth"
    echo "No checkpoint auto-discovered in $EXPERIMENT_DIR/models/ -- falling back to hardcoded iter 60000: $CKPT"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: resolved checkpoint does not exist: $CKPT" >&2
    exit 1
fi

echo "Using checkpoint: $CKPT"

# --- 1. Two-pass ablation (normal vs. all-ref burst-utilization delta) ---
echo "=== Running two-pass burst ablation ==="
torchrun --nproc_per_node=$NPROC analysis/burst_ablation.py \
    --model_path "$CKPT" \
    --mode two_pass \
    --config "$CONFIG" \
    --dataset synburst \
    --data_root "$VAL_ROOT" \
    --num_frames 14

# --- 2. Frame-drop curve (N = 1, 2, 5, 9, 14) ---
echo "=== Running frame-drop ablation ==="
torchrun --nproc_per_node=$NPROC analysis/burst_ablation.py \
    --model_path "$CKPT" \
    --mode frame_drop \
    --drop_counts 1,2,5,9,14 \
    --config "$CONFIG" \
    --dataset synburst \
    --data_root "$VAL_ROOT" \
    --num_frames 14

echo "Done. Logs and plots in analysis/outputs/ablation_logs/"
