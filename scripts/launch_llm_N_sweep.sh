#!/bin/bash
# =============================================================================
# launch_llm_N_sweep.sh  —  Submit the LLM N sweep pipeline to Slurm
# =============================================================================
#
# Submits:
#   1. Job array for N sweep  (4 tasks, indices 5-8, one per N value)
#   2. Plot job for N sweep   (runs after all 4 tasks complete)
#
# The array worker is submit_llm_sweep.sh, which maps indices 0-4 to
# the sigma sweep and indices 5-8 to the N sweep.  This launcher submits
# only the N slice (--array=5-8).
#
# Usage
# -----
#   bash launch_llm_N_sweep.sh
#
# =============================================================================

# =============================================================================
# CONFIG — edit to match your environment
# =============================================================================
HOME_DIR=..
OUT_DIR=${HOME_DIR}/outputs/llm_sweep/quantization
N_SWEEP_DIR=$OUT_DIR/exp2_N_sweep
PARTITION=gpu
GRES=gpu:h200:1
MEM=32G
TIME_SWEEP=3:00:00      # time limit per N value job
                        # N=10 with T=30 takes ~30 min; generous headroom
TIME_PLOT=0:20:00       # time limit for plot job (CPU only)

mkdir -p logs

# =============================================================================
# 1. Submit N sweep array (indices 5-8 for 4 N values)
# =============================================================================
echo "Submitting LLM N sweep array (4 tasks, indices 5-8)..."

N_JOB=$(sbatch \
    --partition=$PARTITION \
    --gres=$GRES \
    --mem=$MEM \
    --time=$TIME_SWEEP \
    --array=5-8 \
    --job-name=llm_N_sweep \
    --output=logs/llm_N_%A_%a.out \
    --error=logs/llm_N_%A_%a.err \
    --parsable \
    submit_llm_sweep.sh)

echo "Submitted LLM N sweep array:     job ID $N_JOB"

# =============================================================================
# 2. Submit N plot job (depends on all N array tasks completing)
# =============================================================================
N_PLOT_JOB=$(sbatch \
    --partition=short \
    --mem=4G \
    --time=$TIME_PLOT \
    --job-name=llm_N_plot \
    --output=logs/llm_N_plot_%j.out \
    --error=logs/llm_N_plot_%j.err \
    --dependency=afterok:$N_JOB \
    --parsable \
    --wrap="python -u ${HOME_DIR}/plot_sweep_summary.py \
        --experiment N \
        --sweep_dir  $N_SWEEP_DIR \
        --out_dir    $N_SWEEP_DIR/summary")

echo "Submitted LLM N plot job:        job ID $N_PLOT_JOB (depends on $N_JOB)"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Output directories:"
echo "  N results : $N_SWEEP_DIR"
echo "  N plots   : $N_SWEEP_DIR/summary"
