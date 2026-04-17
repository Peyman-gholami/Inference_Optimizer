#!/bin/bash
# =============================================================================
# run_fill_table3.sh
# Fills Table 3 and generates the compact reproducibility parameter table.
#
# Edit the scenario blocks below to point to your experiment output dirs.
# Use "--" for any compression method that was not run.
#
# Usage:  bash run_fill_table3.sh
# =============================================================================

HOME_DIR=..
OUT_FILE="${HOME_DIR}/outputs/tables/table3_filled_resnet.tex"

mkdir -p "$(dirname ${OUT_FILE})"

python ${HOME_DIR}/fill_table3.py \
\
    --scenario "OF-Vision-TGH" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/tight/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/tight/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/tight/llmint8" \
\
    --scenario "OF-Vision-TRS" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional/llmint8" \
\
    --scenario "OF-Vision-LOO" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/loose/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/loose/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/loose/llmint8" \
\
    --scenario "OF-Vision-TRk5" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_Rk5/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_Rk5/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_Rk5/llmint8" \
\
    --scenario "OF-Vision-TRK20" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_Rk20/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_Rk20/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_Rk20/llmint8" \
\
    --scenario "OF-Vision-TRK35" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_Rk35/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_Rk35/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_Rk35/llmint8" \
\
    --scenario "OF-Vision-MU05" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu0p5/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_mu0p5/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu0p5/llmint8" \
\
    --scenario "OF-Vision-MU10" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu10/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_mu10/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu10/llmint8" \
\
    --scenario "OF-Vision-MU30" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu30/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_mu30/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_mu30/llmint8" \
\
    --scenario "OF-Vision-EM0" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_etamin0/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_etamin0/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_etamin0/llmint8" \
\
    --scenario "OF-Vision-EM00625" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_etamin0p0625/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_etamin0p0625/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet/transitional_etamin0p0625/llmint8" \
\
    --scenario "OF-Vision-EM025" \
    --topk  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_etamin0p25/topk" \
    --quant "${HOME_DIR}/outputs/single_experiments/group2/resnet_new_quant/transitional_etamin0p25/quantization" \
    --int8  "${HOME_DIR}/outputs/single_experiments/group2/resnet_extra/transitional_etamin0p25/llmint8" \
    --t_idx -1 \
    --out   "${OUT_FILE}"

echo ""
echo "Output written to: ${OUT_FILE}"