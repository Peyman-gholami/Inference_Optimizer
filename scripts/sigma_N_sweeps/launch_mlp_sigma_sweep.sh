#!/bin/bash
# =============================================================================
# launch_mlp_sigma_sweep.sh  —  Submit the MLP sigma sweep pipeline to Slurm
# =============================================================================
#
# Submits:
#   1. Job array for sigma sweep  (5 tasks, one per sigma value)
#   2. Plot job for sigma sweep   (runs after all sigma tasks complete)
#
# Usage
# -----
#   bash launch_mlp_sigma_sweep.sh
#
# =============================================================================

# =============================================================================
# CONFIG — edit to match your environment
# =============================================================================
HOME_DIR=..
COMPRESSOR_TYPE=topk            # must match COMPRESSOR_TYPE in submit_mlp_sweep.sh
OUT_DIR=${HOME_DIR}/outputs/mlp_sweep/${COMPRESSOR_TYPE}
SIGMA_SWEEP_DIR=$OUT_DIR/exp1_sigma_sweep
PARTITION=gpu
GRES=gpu:h200:1
MEM=8G
TIME_SWEEP=02:00:00     # time limit per sigma value job
TIME_PLOT=0:15:00       # time limit for plot job (CPU only)

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
    --job-name=mlp_sigma_sweep \
    --output=logs/mlp_sigma_%A_%a.out \
    --error=logs/mlp_sigma_%A_%a.err \
    --parsable \
    submit_mlp_sweep.sh sigma)

echo "Submitted MLP sigma sweep array: job ID $SIGMA_JOB"

# =============================================================================
# 2. Submit sigma plot job (depends on all sigma array tasks completing)
# =============================================================================
SIGMA_PLOT_JOB=$(sbatch \
    --partition=short \
    --mem=4G \
    --time=$TIME_PLOT \
    --job-name=mlp_sigma_plot \
    --output=logs/mlp_sigma_plot_%j.out \
    --error=logs/mlp_sigma_plot_%j.err \
    --dependency=afterok:$SIGMA_JOB \
    --parsable \
    --wrap="python -u ${HOME_DIR}/plot_sweep_summary.py \
        --experiment sigma \
        --sweep_dir  $SIGMA_SWEEP_DIR \
        --out_dir    $SIGMA_SWEEP_DIR/summary")

echo "Submitted MLP sigma plot job:    job ID $SIGMA_PLOT_JOB (depends on $SIGMA_JOB)"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Output directories:"
echo "  Sigma results : $SIGMA_SWEEP_DIR"
echo "  Sigma plots   : $SIGMA_SWEEP_DIR/summary"
