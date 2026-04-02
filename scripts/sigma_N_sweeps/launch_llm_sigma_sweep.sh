#!/bin/bash
# =============================================================================
# launch_llm_sigma_sweep.sh  —  Submit the LLM sigma sweep pipeline to Slurm
# =============================================================================
#
# Submits:
#   1. Job array for sigma sweep  (5 tasks, indices 0-4, one per sigma value)
#   2. Plot job for sigma sweep   (runs after all 5 tasks complete)
#
# The array worker is submit_llm_sweep.sh, which maps indices 0-4 to
# the sigma sweep and indices 5-8 to the N sweep.  This launcher submits
# only the sigma slice (--array=0-4).
#
# Usage
# -----
#   bash launch_llm_sigma_sweep.sh
#
# =============================================================================

# =============================================================================
# CONFIG — edit to match your environment
# =============================================================================
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/llm_sweep/quantization
SIGMA_SWEEP_DIR=$OUT_DIR/exp1_sigma_sweep
PARTITION=gpu
GRES=gpu:h200:1
MEM=32G
TIME_SWEEP=3:00:00      # time limit per sigma value job
TIME_PLOT=0:20:00       # time limit for plot job (CPU only)

mkdir -p logs

# =============================================================================
# 1. Submit sigma sweep array (indices 0-4 for 5 sigma values)
# =============================================================================
echo "Submitting LLM sigma sweep array (5 tasks, indices 0-4)..."

SIGMA_JOB=$(sbatch \
    --partition=$PARTITION \
    --gres=$GRES \
    --mem=$MEM \
    --time=$TIME_SWEEP \
    --array=0-4 \
    --job-name=llm_sigma_sweep \
    --output=logs/llm_sigma_%A_%a.out \
    --error=logs/llm_sigma_%A_%a.err \
    --parsable \
    submit_llm_sweep.sh)

echo "Submitted LLM sigma sweep array: job ID $SIGMA_JOB"

# =============================================================================
# 2. Submit sigma plot job (depends on all sigma array tasks completing)
# =============================================================================
SIGMA_PLOT_JOB=$(sbatch \
    --partition=short \
    --mem=4G \
    --time=$TIME_PLOT \
    --job-name=llm_sigma_plot \
    --output=logs/llm_sigma_plot_%j.out \
    --error=logs/llm_sigma_plot_%j.err \
    --dependency=afterok:$SIGMA_JOB \
    --parsable \
    --wrap="python -u ${HOME_DIR}/plot_sweep_summary.py \
        --experiment sigma \
        --sweep_dir  $SIGMA_SWEEP_DIR \
        --out_dir    $SIGMA_SWEEP_DIR/summary")

echo "Submitted LLM sigma plot job:    job ID $SIGMA_PLOT_JOB (depends on $SIGMA_JOB)"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Output directories:"
echo "  Sigma results : $SIGMA_SWEEP_DIR"
echo "  Sigma plots   : $SIGMA_SWEEP_DIR/summary"
