#!/bin/bash
# =============================================================================
# run_fill_table3_llama.sh
# Fills Table 3 and generates the compact reproducibility parameter table.
#
# Edit the scenario blocks below to point to your experiment output dirs.
# Use "--" for any compression method that was not run.
#
# Usage:  bash run_fill_table3_llama.sh
# =============================================================================

HOME_DIR=..
OUT_FILE="${HOME_DIR}/outputs/tables/table3_filled_llama.tex"

mkdir -p "$(dirname ${OUT_FILE})"

python ${HOME_DIR}/fill_table3.py \
\
    --scenario "OF-Lang-SM" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/llama/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/llama/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/llama/llmint8" \
\
    --scenario "OF-Lang-EXa" \
    --topk "${HOME_DIR}/outputs/single_experiments/group2_extra_a/llama/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_extra_a/llama/quantization" \
    --int8 "${HOME_DIR}/outputs/single_experiments/group2_extra_a/llama/llmint8_mu0001" \
\
    --scenario "OF-Lang-RR1" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2_rerun1/llama/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_rerun1/llama/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2_rerun1/llama/llmint8" \
\
    --scenario "OF-Lang-RR2b" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2_rerun2b/llama/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_rerun2b/llama/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2_rerun2b/llama/llmint8" \
    --t_idx -1 \
    --out   "${OUT_FILE}"

echo ""
echo "Output written to: ${OUT_FILE}"