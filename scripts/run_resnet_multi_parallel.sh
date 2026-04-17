#!/bin/bash
# =============================================================================
# run_resnet_multi_parallel.sh
#
# Submits multi-task ResNet experiments to populate Table 4 (multi-task
# offline results) and the corresponding reproducibility table.
#
# Each "scenario" is a named experimental configuration.  For each scenario
# we run three compressors (topk, quantization, llmint8) in parallel.
#
# Scenarios:
#
#   MT-R1  K=2, symmetric tasks (identical a_i, equal weights)
#          Baseline verification: water-filling should split evenly.
#
#   MT-R2  K=2, heterogeneous a_i (different cutpoints → different payload)
#          Key experiment: asymmetric allocation driven by payload size.
#
#   MT-R3  K=3, tight channel (all tasks competing for scarce bandwidth)
#          Contention stress test: some tasks must compress aggressively.
#
#   MT-R4  K=2, asymmetric weights (w_1=1, w_2=3)
#          Priority experiment: high-weight task gets more bandwidth.
#
#   MT-R5  K=2, loose channel (verify near-uncompressed at high capacity)
#          Easy regime: CSI-aware should push both tasks to eta≈1.
#
# Usage:   bash run_resnet_multi_parallel.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# SHARED PARAMETERS
# =============================================================================

T=100
SEED=12
MU=3.0
EPSILON=0.1
ETA_MIN=0.125
SIGMA=0.05
N=20

OUT_BASE="${HOME_DIR}/outputs/multi_experiments/resnet"

TIME_LIMIT="3:00:00"
MEM="16G"

# =============================================================================
# END OF SHARED PARAMETERS
# =============================================================================

mkdir -p logs

# ─────────────────────────────────────────────────────────────────────────────
# submit_multi SUFFIX K C_MIN C_MAX R_K EXTRA_ARGS
#   Submits topk + quantization + llmint8 for one scenario.
#   EXTRA_ARGS: optional additional flags (e.g. --heterogeneous_a false)
# ─────────────────────────────────────────────────────────────────────────────
submit_multi() {
    local SUFFIX=$1
    local K=$2
    local C_MIN=$3
    local C_MAX=$4
    local R_K=$5
    shift 5
    local EXTRA_ARGS="$*"

    local OUT_BASE_SCEN="${OUT_BASE}/${SUFFIX}"

    echo ""
    echo "Submitting: ${SUFFIX}  K=${K}  c~U[${C_MIN},${C_MAX}]  R_k=${R_K}"
    echo "  extra: ${EXTRA_ARGS}"
    echo "  out:   ${OUT_BASE_SCEN}/{topk,quantization,llmint8}"

    for COMP in topk quantization llmint8; do
        JOB=$(sbatch --parsable \
            --partition=gpu \
            --gres=gpu:h200:1 \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 \
            --mem=${MEM} \
            --time=${TIME_LIMIT} \
            --job-name=mt_${SUFFIX}${COMP} \
            --output=logs/mt${SUFFIX}${COMP}%j.out \
            --error=logs/mt_${SUFFIX}${COMP}%j.err \
            --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=8,14,21
export RESNET_FAST_BATCHES=1
export ETA_MIN0=${ETA_MIN}
export ETA_MIN1=${ETA_MIN}
export RESNET_GRAD_SIGMA=${SIGMA}
export RESNET_GRAD_N=${N}
export COMPRESSOR_TYPE=${COMP}
export COMPRESSOR_BACKEND=gpu

python ${HOME_DIR}/run_experiment_multi.py \\
    --backend resnet \\
    --K ${K} \\
    --compressor ${COMP} \\
    --compressor_backend gpu \\
    --eta_min ${ETA_MIN} \\
    --no_csi_mu ${MU} \\
    --no_csi_epsilon ${EPSILON} \\
    --R_k ${R_K} \\
    --c_min ${C_MIN} --c_max ${C_MAX} \\
    --T ${T} --seed ${SEED} \\
    --out_dir ${OUT_BASE_SCEN}/${COMP} \\
    ${EXTRA_ARGS}
")
        echo "  Submitted ${COMP}: job ${JOB}"
    done
}

# =============================================================================
# EXPERIMENTS
# =============================================================================

# ── MT-R1: K=2 symmetric, transitional channel ───────────────────────────────
# Both tasks identical. Water-filling should split equally.
# Feasibility: a_eff=65536, eta_min=0.125, R_k=10 → need c >= 81920 B/s ✓
#submit_multi "MT-R1-sym" 2 \
#    60000 300000 10

# ── MT-R2: K=2 heterogeneous a_i (built-in via --heterogeneous_a true) ───────
# Different payload sizes per task → asymmetric bandwidth allocation.
# heterogeneous_a scales a_{i,k} differently per task by default.
#submit_multi "MT-R2-hetero" 2 \
#    60000 300000 10 \
#    --heterogeneous_a true

# ── MT-R3: K=3 tight channel, high contention ────────────────────────────────
# Three tasks on the same links. Channel is tighter (R_k=10, c_min=40k).
# Some tasks will be forced to compress aggressively.
#submit_multi "MT-R3-K3-tight" 3 \
#    40000 200000 10

# ── MT-R4: K=2 asymmetric weights (w_1=1.0, w_2=3.0) ────────────────────────
# High-weight task 2 should receive preferential bandwidth allocation.
# Tests the priority-aware water-filling behaviour.
submit_multi "MT-R4-weights" 2 \
    60000 300000 10 \
    --task_weights 1.0 3.0

# ── MT-R5: K=2 loose channel (easy regime) ───────────────────────────────────
# Very high capacity: both tasks should operate near eta=1 (no compression).
# CSI-aware should achieve near-maximum accuracy for all methods.
#submit_multi "MT-R5-loose" 2 \
#    200000 600000 10

# ── MT-R6: K=2 tight R_k (stricter deadline) ─────────────────────────────────
# Same channel as MT-R1 but R_k=20. Feasibility boundary at c=163,840.
# With c_min=60k, ~35% of slots infeasible → tests infeasibility handling.
#submit_multi "MT-R6-tight-Rk" 2 \
#    60000 300000 20

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Output structure:"
echo "  ${OUT_BASE}/MT-R1-sym/{topk,quantization,llmint8}/"
echo "  ${OUT_BASE}/MT-R2-hetero/{topk,quantization,llmint8}/"
echo "  ...etc."
echo ""
echo "To generate LaTeX tables after all jobs complete:"
echo "  bash run_fill_table4_resnet.sh"