#!/bin/bash
# =============================================================================
# launch_resnet_sigma_sweep.sh  —  Submit the full sweep pipeline to Slurm
# =============================================================================
#
# Submits:
#   1. Job array for N sweep     (4 tasks, one per N value)
#   2. Plot job for N sweep      (runs after all N tasks complete)
#
# Usage
# -----
#   bash launch_resnet_sigma_sweep.sh
#
# =============================================================================

# =============================================================================
# CONFIG — edit to match your environment
# =============================================================================
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/resnet_sweep/llmint8 # valid values topk (default), quantization, llmint8
SIGMA_SWEEP_DIR=$OUT_DIR/exp1_sigma_sweep
N_SWEEP_DIR=$OUT_DIR/exp2_N_sweep
PARTITION=gpu
GRES=gpu:h200:1
MEM=16G
TIME_SWEEP=4:00:00      # time limit per sweep value
TIME_PLOT=0:30:00       # time limit for plot job (CPU only)

mkdir -p logs

# =============================================================================
# 1. Submit sigma sweep array (indices 0-4 for 5 sigma values)
# =============================================================================
SIGMA_JOB=$(sbatch \
    --partition=$PARTITION \
    --gres=$GRES \
    --mem=$MEM \
    --time=$TIME_SWEEP \
    --array=0-4 \
    --job-name=sigma_sweep \
    --output=logs/sigma_%A_%a.out \
    --error=logs/sigma_%A_%a.err \
    --parsable \
    submit_resnet_sweep.sh sigma)

echo "Submitted sigma sweep array: job ID $SIGMA_JOB"

# =============================================================================
# 2. Submit sigma plot job (depends on all sigma array tasks completing)
# =============================================================================
SIGMA_PLOT_JOB=$(sbatch \
    --partition=short \
    --mem=4G \
    --time=$TIME_PLOT \
    --job-name=sigma_plot \
    --output=logs/sigma_plot_%j.out \
    --error=logs/sigma_plot_%j.err \
    --dependency=afterok:$SIGMA_JOB \
    --parsable \
    --wrap="python -u ${HOME_DIR}/plot_sweep_summary.py \
        --experiment sigma \
        --sweep_dir  $SIGMA_SWEEP_DIR \
        --out_dir    $SIGMA_SWEEP_DIR/summary")

echo "Submitted sigma plot job:    job ID $SIGMA_PLOT_JOB (depends on $SIGMA_JOB)"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Output directories:"
echo "  Sigma results : $SIGMA_SWEEP_DIR"
echo "  Sigma plots   : $SIGMA_SWEEP_DIR/summary"
