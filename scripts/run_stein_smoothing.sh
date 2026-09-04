#!/bin/bash
# =============================================================================
# run_stein_smoothing.sh
#
# Measures the function the Stein oracle actually sees: how far the
# Gaussian-smoothed A_sigma(eta) = E_z[A(eta + sigma z)] sits from A, and what
# the gap costs the controller. Produces the JSON that fill_table_stein.py
# turns into the Stein-path analogue of appendix Table 7.
#
# One job per compressor, submitted in parallel.
#
# Usage:  cd scripts && bash run_stein_smoothing.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS
# =============================================================================

N_GRID=6            # points per eta axis (total = N_GRID^3 evaluation points)
N_DRAWS=200         # MC draws per (eta, sigma) for A_sigma
SIGMAS="0.01,0.02,0.05,0.1,0.2"   # 0.05 is the value used in the experiments
FAST_BATCHES=2      # batches per fast eval (2 x 100 = 200 images)
GRAD_N=20           # antithetic pairs, matching V-TRS / V-LOO
GRAD_SEEDS=8        # seeds for the gradient-stability probe
GRAD_POINTS=5

# eta ladder for the codecs that snap (quantization: 0.0625 INT2, 0.125 INT4,
# 0.25 INT8, 0.5 FP16, anything else passthrough). A uniform grid lands between
# rungs, where the codec is a no-op, and measures nothing.
QUANT_ETA_AXIS="0.0625,0.125,0.25,0.5,1.0"

CUTPOINTS=8,14,21   # 3 links -> a = [0.066, 0.033, 0.016] MB (V-TRS / V-LOO)

TIME_RESNET="3:00:00"
MEM="32G"

OUT_BASE="${HOME_DIR}/outputs/stein_smoothing"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

# ── ResNet ────────────────────────────────────────────────────────────────────
submit_resnet() {
    local COMP=$1
    local EXTRA_FLAGS="${2:-}"   # optional extra flags e.g. --eta_axis ...
    local OUT_DIR="${OUT_BASE}/resnet_${COMP}"

    echo "Submitting Stein smoothing: compressor=${COMP} ${EXTRA_FLAGS}"
    echo "  out: ${OUT_DIR}"

    JOB=$(sbatch --parsable \
        --partition=gpu \
        --gres=gpu:h200:1 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=4 \
        --mem=${MEM} \
        --time=${TIME_RESNET} \
        --job-name=stein_smooth_${COMP} \
        --output=logs/stein_smooth_${COMP}_%j.out \
        --error=logs/stein_smooth_${COMP}_%j.err \
        --wrap="
set -e
echo \"[wrap] Job \$SLURM_JOB_ID started on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate newcap

export INFERENCE_OPTIMIZER_BACKEND=resnet
export RESNET_CHECKPOINT=${HOME_DIR}/models/resnet56-4bfd9763.th
export RESNET_DATA_ROOT=${HOME_DIR}/data
export RESNET_CUTPOINTS=${CUTPOINTS}
export COMPRESSOR_TYPE=${COMP}
export COMPRESSOR_BACKEND=gpu

python ${HOME_DIR}/stein_smoothing_table.py \\
    --compressor ${COMP} \\
    --compressor_backend gpu \\
    --scenario 'V-TRS/V-LOO' \\
    --task RN56-CF10 \\
    --cutpoints ${CUTPOINTS} \\
    --n_grid ${N_GRID} \\
    --n_draws ${N_DRAWS} \\
    --sigmas ${SIGMAS} \\
    --fast_batches ${FAST_BATCHES} \\
    --grad_N ${GRAD_N} \\
    --grad_seeds ${GRAD_SEEDS} \\
    --grad_points ${GRAD_POINTS} \\
    --out_dir ${OUT_DIR} \\
    ${EXTRA_FLAGS}
")
    echo "  Submitted: job ${JOB}"
}

# =============================================================================
# SUBMIT
# =============================================================================

echo "=== ResNet Stein smoothing ==="
submit_resnet topk
submit_resnet quantization "--eta_axis ${QUANT_ETA_AXIS}"
#submit_resnet llmint8 "--eta_axis ${QUANT_ETA_AXIS}"   # skip — flat accuracy landscape

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be in:"
echo "  ${OUT_BASE}/resnet_topk/stein_smoothing_results.json"
echo "  ${OUT_BASE}/resnet_quantization/stein_smoothing_results.json"
echo ""
echo "Each log also prints the table as plain text when the job finishes."
echo ""
echo "To build the LaTeX table (no GPU needed, reads the JSON):"
echo "  python ${HOME_DIR}/fill_table_stein.py \\"
echo "      --run ${OUT_BASE}/resnet_topk \\"
echo "      --run ${OUT_BASE}/resnet_quantization \\"
echo "      --out ${HOME_DIR}/../MobiHoc_2026/tables/stein_smoothing_table.tex"
echo ""
echo "Note: quantization snaps eta to a ladder and is passthrough between"
echo "rungs, so a Gaussian perturbation almost never lands on a supported"
echo "level. Expect A_sigma to be flat in sigma there and the Stein gradient"
echo "to be identically zero — that is the result, not a failed run."
