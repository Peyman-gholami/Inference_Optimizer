#!/bin/bash
# =============================================================================
# run_fill_table7.sh
# Fills Table 7 and generates the compact reproducibility parameter table.
#
# Usage:  bash run_fill_table7.sh
# =============================================================================

HOME_DIR=..
#OUT_FILE="${HOME_DIR}/outputs/tables/table3_filled_resnet.tex"

#mkdir -p "$(dirname ${OUT_FILE})"

python ${HOME_DIR}/generate_latex_table_single.py \
      --out_dir "${HOME_DIR}/outputs/single_experiments/group2/resnet/tight/topk" \
      --label "OF-V
      
      
      
      t" \
      --repro_row \
      --t_idx -1

