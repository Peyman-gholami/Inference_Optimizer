#!/bin/bash
# =============================================================================
# launch_mlp_N_sweep.sh  —  Submit the MLP N sweep pipeline to Slurm
# =============================================================================
#
# Submits:
#   1. Job array for N sweep  (5 tasks, one per N value)
#   2. Plot job for N sweep   (runs after all N tasks complete)
#
# Usage
# -----
#   bash launch_mlp_N_sweep.sh
#
# =============================================================================

# =============================================================================
# CONFIG — edit to match your environment
# =============================================================================
HOME_DIR=..
COMPRESSOR_TYPE=topk            # must match COMPRESSOR_TYPE in submit_mlp_sweep.sh
OUT_DIR=${HOME_DIR}/outputs/mlp_sweep/${COMPRESSOR_TYPE}
N_SWEEP_DIR=$OUT_DIR/exp2_N_sweep
PARTITION=gpu
GRES=gpu:h200:1
MEM=8G
TIME_SWEEP=02:00:00     # time limit per N value job
                        # N=100 with T=150 is still fast (~15 min)
TIME_PLOT=0:15:00       # time limit for plot job (CPU only)

mkdir -p logs

# =============================================================================
# 1. Submit N sweep array (indices 0-4 for 5 N values)
# =============================================================================
N_JOB=$(sbatch \
    --partition=$PARTITION \
    --gres=$GRES \
    --mem=$MEM \
    --time=$TIME_SWEEP \
    --array=0-4 \
    --job-name=mlp_N_sweep \
    --output=logs/mlp_N_%A_%a.out \
    --error=logs/mlp_N_%A_%a.err \
    --parsable \
    submit_mlp_sweep.sh N)

echo "Submitted MLP N sweep array:     job ID $N_JOB"

# =============================================================================
# 2. Submit N plot job (depends on all N array tasks completing)
# =============================================================================
N_PLOT_JOB=$(sbatch \
    --partition=short \
    --mem=4G \
    --time=$TIME_PLOT \
    --job-name=mlp_N_plot \
    --output=logs/mlp_N_plot_%j.out \
    --error=logs/mlp_N_plot_%j.err \
    --dependency=afterok:$N_JOB \
    --parsable \
    --wrap="python -u ${HOME_DIR}/plot_sweep_summary.py \
        --experiment N \
        --sweep_dir  $N_SWEEP_DIR \
        --out_dir    $N_SWEEP_DIR/summary")

echo "Submitted MLP N plot job:        job ID $N_PLOT_JOB (depends on $N_JOB)"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Output directories:"
echo "  N results : $N_SWEEP_DIR"
echo "  N plots   : $N_SWEEP_DIR/summary"
