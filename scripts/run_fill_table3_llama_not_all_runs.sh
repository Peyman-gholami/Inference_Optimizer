#!/bin/bash
# =============================================================================
# run_fill_table3_llama_not_all_runs.sh
# Fills Table 3 and generates the compact reproducibility parameter table.
#
# Edit the scenario blocks below to point to your experiment output dirs.
# Use "--" for any compression method that was not run.
#
# Usage:  bash run_fill_table3_llama_not_all_runs.sh
# =============================================================================

HOME_DIR=..
OUT_FILE="${HOME_DIR}/outputs/tables/table3_filled_llama_with_predictors.tex"

mkdir -p "$(dirname ${OUT_FILE})"

python ${HOME_DIR}/fill_table3_not_all_runs.py \
\
    --scenario "OF-Lang-PR025-R1" \
    --topk  "empty" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_predictor_025/llama/quantization" \
    --int8  "empty" \
\
    --scenario "OF-Lang-PR025-R2" \
    --topk "${HOME_DIR}/outputs/single_experiments/group2_predictor_025/llama/topk_per_token" \
    --quant  "empty" \
    --int8  "empty" \
\
    --scenario "OF-Lang-PR00-R1" \
    --topk  "empty" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_predictor_00/run1/quantization" \
    --int8  "empty" \
\
    --scenario "OF-Lang-PR00-R2" \
    --topk "${HOME_DIR}/outputs/single_experiments/group2_predictor_00/run2/topk_per_token" \
    --quant  "empty" \
    --int8  "empty" \
\
    --scenario "OF-Lang-PR00-R3" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2_predictor_00/run3/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_predictor_00/run3/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2_predictor_00/run3/llmint8" \
\
    --scenario "OF-Lang-PR01-5shots-R1" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2_predictor/run1/topk_per_token" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2_predictor/run1/quantization" \
    --int8 "${HOME_DIR}/outputs/single_experiments/group2_predictor/run1/llmint8" \
    --t_idx -1 \
    --out   "${OUT_FILE}"

echo ""
echo "Output written to: ${OUT_FILE}"