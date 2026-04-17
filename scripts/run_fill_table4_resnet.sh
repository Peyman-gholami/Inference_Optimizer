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
    --scenario "OF-Vision-TGH" \
    --topk  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/topk" \
    --quant "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/quantization" \
    --int8  "${HOME_DIR}/outputs/multi_experiments/resnet/MT-R1-sym/llmint8" \
    --t_idx -1 \
    --out   "${OUT_FILE}"

echo ""
echo "Output written to: ${OUT_FILE}"