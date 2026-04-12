#!/bin/bash
# =============================================================================
# run_resnet_parallel.sh
#
# Submits one Slurm job per compression method (topk, quantization, llmint8)
# all running in parallel with the same experiment parameters.
#
# Edit the PARAMETERS section below to configure the experiment.
# Each call to submit_experiment() launches 3 parallel jobs.
#
# Usage:  bash run_resnet_parallel.sh
# =============================================================================

HOME_DIR=..

# =============================================================================
# PARAMETERS — edit these for each experiment group
# =============================================================================

# Channel regime
C_MIN=60000
C_MAX=300000

# Throughput demand — higher R_k = stricter deadline = harder problem
# Feasibility boundary: c[i] >= a[i] * eta_min * R_k
# ResNet a=[65536,32768,16384], eta_min=0.125:
#   R_k=5  → boundary = 40,960  (easy)
#   R_k=10 → boundary = 81,920  (transitional)
#   R_k=20 → boundary = 163,840 (tighter)
#   R_k=50 → boundary = 409,600 (very tight)
R_K=10

# Horizon
T=50

# Gradient oracle
SIGMA=0.05
GRAD_N=20

# No-CSI
MU=3.0
EPSILON=0.1

# eta_min (per link)
ETA_MIN=0.125

# Output directory prefix — compression method appended automatically
OUT_PREFIX="${HOME_DIR}/outputs/single_experiments/group2/resnet"

# Slurm
TIME_LIMIT="2:00:00"
MEM="8G"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

mkdir -p logs

submit_experiment() {
    local SUFFIX=$1
    local OUT_BASE="${OUT_PREFIX}/${SUFFIX}"

    echo ""
    echo "Submitting experiment: ${SUFFIX}"
    echo "  c_min=${C_MIN}  c_max=${C_MAX}  R_k=${R_K}  T=${T}"
    echo "  sigma=${SIGMA}  N=${GRAD_N}  mu=${MU}  eta_min=${ETA_MIN}"
    echo "  out: ${OUT_BASE}/{topk,quantization,llmint8}"

    for COMP in topk quantization llmint8; do
        JOB=$(sbatch --parsable \
            --partition=gpu \
            --gres=gpu:h200:1 \
            --nodes=1 --ntasks=1 \
            --cpus-per-task=4 \
            --mem=${MEM} \
            --time=${TIME_LIMIT} \
            --job-name=resnet_${SUFFIX}_${COMP} \
            --output=logs/resnet_${SUFFIX}_${COMP}_%j.out \
            --error=logs/resnet_${SUFFIX}_${COMP}_%j.err \
            --wrap="
mkdir -p logs
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
export RESNET_GRAD_N=${GRAD_N}
export COMPRESSOR_TYPE=${COMP}
export COMPRESSOR_BACKEND=gpu

python ${HOME_DIR}/run_experiment.py --backend resnet \
    --compressor ${COMP} \
    --compressor_backend gpu \
    --eta_min ${ETA_MIN} \
    --no_csi_mu ${MU} \
    --no_csi_epsilon ${EPSILON} \
    --R_k ${R_K} \
    --c_min ${C_MIN} --c_max ${C_MAX} --T ${T} \
    --out_dir ${OUT_BASE}/${COMP}
")
        echo "  Submitted ${COMP}: job ${JOB}"
    done
}

# =============================================================================
# EXPERIMENTS — uncomment / add submit_experiment calls as needed.
#
# Pattern: set parameters, then call submit_experiment "descriptive_suffix"
# The suffix becomes part of the output directory name.
# =============================================================================

# ── Baseline: transitional regime, default parameters ────────────────────────
#submit_experiment "transitional"

# ── Channel regime ────────────────────────────────────────────────────────────
C_MIN=40000;  C_MAX=100000;  R_K=10; submit_experiment "tight"
#C_MIN=200000; C_MAX=600000;  R_K=10; submit_experiment "loose"

# ── R_k sensitivity (same channel, different throughput demand) ───────────────
# Increasing R_k moves the feasibility boundary up — same channel becomes tight.
#C_MIN=60000; C_MAX=300000
#R_K=5;  submit_experiment "transitional_Rk5"
#R_K=20; submit_experiment "transitional_Rk20"
#R_K=35; submit_experiment "transitional_Rk35"

# ── μ sensitivity ─────────────────────────────────────────────────────────────
#C_MIN=60000; C_MAX=300000; R_K=10
#MU=0.5;  submit_experiment "transitional_mu0p5"
#MU=10.0; submit_experiment "transitional_mu10"
#MU=30.0; submit_experiment "transitional_mu30"

# ── η_min sensitivity ─────────────────────────────────────────────────────────
#C_MIN=60000; C_MAX=300000; R_K=10; MU=3.0
#ETA_MIN=0.0;    submit_experiment "transitional_etamin0"
#ETA_MIN=0.0625; submit_experiment "transitional_etamin0p0625"
#ETA_MIN=0.25;   submit_experiment "transitional_etamin0p25"
#ETA_MIN=0.5;    submit_experiment "transitional_etamin0p5"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
