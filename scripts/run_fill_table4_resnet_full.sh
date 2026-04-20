#!/bin/bash
# =============================================================================
# run_fill_trun_fill_table4_resnetable3.sh
# Fills Table 3 and generates the compact reproducibility parameter table.
#
# Edit the scenario blocks below to point to your experiment output dirs.
# Use "--" for any compression method that was not run.
#
# Usage:  bash run_fill_table4_resnet.sh
# =============================================================================

HOME_DIR=..
OUT_FILE="${HOME_DIR}/outputs/tables/table4_filled_resnet.tex"

mkdir -p "$(dirname ${OUT_FILE})"

python ${HOME_DIR}/fill_table4_resnet.py \
\
    --scenario "MT-R1-Sym" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/llmint8" \
\
    --scenario "MT-R2-Hetero" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R2-hetero/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R2-hetero/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R2-hetero/llmint8" \
\
    --scenario "MT-R3-K3-Tight" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R3-K3-tight/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R3-K3-tight/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R3-K3-tight/llmint8" \
\
    --scenario "MT-R4-weights" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R4-weights/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R4-weights/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R4-weights/llmint8" \
\
    --scenario "MT-R5-loose-N10-T100" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose/llmint8" \
\
    --scenario "MT-R5-loose-N10-T50" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose-N10-T50/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose-N10-T50/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R5-loose-N10-T50/llmint8" \
\
    --scenario "MT-R6-tight-Rk-N10-T100" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk/llmint8" \
\
    --scenario "MT-R6-tight-Rk-N10-T50" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk-N10-T50/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk-N10-T50/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R6-tight-Rk-N10-T50/llmint8" \
    --t_idx -1 \
    --out   "${OUT_FILE}"

echo ""
echo "Output written to: ${OUT_FILE}"