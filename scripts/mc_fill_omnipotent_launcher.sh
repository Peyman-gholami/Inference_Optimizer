#!/bin/bash
# =============================================================================
# mc_fill_omnipotent_launcher.sh
#
# Long-lived poller (run it with nohup/disown, or via `run_in_background`)
# that submits the remaining mc_concavity.py fill chains one job at a time,
# respecting the `gpu` QOS's MaxSubmitPU=8 cap (jobs queued+pending+running
# all count against it). Each task is a 3-link SLURM afterany chain
# (--floor_max 0.7, --min_box_samples 100); a task's next link is only
# submitted once its previous link's JobID is known, and a NEW link (for any
# task) is only submitted once total in-flight jobs for this user drop
# below the cap.
#
# Safe to leave running unattended: cmd_fill resumes from
# <stem>_fill.partial.json and is idempotent (exits immediately, no GPU load,
# once a task's floors are satisfied), so extra/late links are cheap no-ops.
#
# Usage: nohup bash scripts/mc_fill_omnipotent_launcher.sh > logs/mc_fill_launcher.log 2>&1 &
#        disown
# Tail progress: tail -f logs/mc_fill_launcher.log
# =============================================================================
set -u
HOME_DIR=.
MIN_BOX_SAMPLES=100
FLOOR_STEP=0.1
FLOOR_MAX=0.7
TIME="08:00:00"
MEM="48G"
LINKS_PER_TASK=3
CAP=8              # gpu QOS MaxSubmitPU
POLL_SECS=120

mkdir -p logs

PPL_EXTRA="--max_texts 128 --max_length 512 --batch_size 1"
ACC_EXTRA="--samples_per_subject 20 --n_shot 5 --max_length 512 --batch_size 16"
OUT=outputs/mc_concavity

# name|model|dataset|metric|strategy|rays_json|out_dir|tol|extra
TASKS=(
"gemma2b_topk_c7|google/gemma-2b|sharegpt|perplexity|topk_per_token|${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token/cuts7_merged_rays.json|${OUT}/google_gemma-2b/sharegpt_perplexity/topk_per_token|0.001|${PPL_EXTRA}"
"gemma2b_reserve_c7|google/gemma-2b|sharegpt|perplexity|llmint8_reserve|${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve/cuts7_merged_rays.json|${OUT}/google_gemma-2b/sharegpt_perplexity/llmint8_reserve|0.001|${PPL_EXTRA}"
"gemma7b_topk_c4|google/gemma-7b|sharegpt|perplexity|topk_per_token|${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token/cuts4_merged_rays.json|${OUT}/google_gemma-7b/sharegpt_perplexity/topk_per_token|0.001|${PPL_EXTRA}"
"gemma7b_reserve_c4|google/gemma-7b|sharegpt|perplexity|llmint8_reserve|${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve/cuts4_merged_rays.json|${OUT}/google_gemma-7b/sharegpt_perplexity/llmint8_reserve|0.001|${PPL_EXTRA}"
"llama_wikitext_topk_c4|meta-llama/Llama-3.1-8B|wikitext|perplexity|topk_per_token|${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token/cuts4_merged_rays.json|${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/topk_per_token|0.001|${PPL_EXTRA}"
"llama_wikitext_reserve_c4|meta-llama/Llama-3.1-8B|wikitext|perplexity|llmint8_reserve|${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve/cuts4_merged_rays.json|${OUT}/meta-llama_Llama-3.1-8B/wikitext_perplexity/llmint8_reserve|0.001|${PPL_EXTRA}"
"llama_mmlu_topk_c4|meta-llama/Llama-3.1-8B|mmlu|accuracy|topk_per_token|${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy/topk_merged_rays.json|${OUT}/meta-llama_Llama-3.1-8B/mmlu_accuracy|0.02|${ACC_EXTRA}"
)

# gemma2b_topk_c7 link 1 already submitted as job 8111715 (outside this
# script) — seed its "last job id" so link 2 depends on it instead of
# resubmitting link 1.
declare -A LAST_JOBID
declare -A NEXT_LINK
LAST_JOBID["gemma2b_topk_c7"]="8111715"
NEXT_LINK["gemma2b_topk_c7"]=2
for t in "${TASKS[@]}"; do
    name="${t%%|*}"
    if [ -z "${NEXT_LINK[$name]:-}" ]; then
        NEXT_LINK["$name"]=1
    fi
done

in_flight() {
    squeue -u "$USER" -h 2>/dev/null | wc -l
}

submit_link() {
    local name=$1 model=$2 dataset=$3 metric=$4 strategy=$5 rays_json=$6 \
          out_dir=$7 tol=$8 extra=$9 link=${10} depjob=${11}
    local deparg=()
    [ -n "$depjob" ] && deparg=(--dependency=afterany:${depjob})
    sbatch --parsable \
        --partition=gpu --gres=gpu:a100:1 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=${MEM} --time=${TIME} \
        "${deparg[@]}" \
        --job-name=mcfillauto_${name} \
        --output=logs/mcfillauto_${name}_link${link}_%j.out \
        --error=logs/mcfillauto_${name}_link${link}_%j.err \
        --wrap="
set -e
echo \"[wrap] ${name} link ${link} Job \$SLURM_JOB_ID on \$(hostname) at \$(date)\" >&2
eval \"\$(conda shell.bash hook)\"
conda deactivate
conda activate easy
python ${HOME_DIR}/experiments/mc_concavity.py fill \\
    --rays_json ${rays_json} \\
    --model ${model} --dataset ${dataset} --metric ${metric} --strategy ${strategy} \\
    --tol ${tol} --min_box_samples ${MIN_BOX_SAMPLES} --adaptive_floor_step ${FLOOR_STEP} \\
    --floor_max ${FLOOR_MAX} \\
    ${extra} \\
    --out_dir ${out_dir}"
}

echo "[$(date)] launcher started. TASKS=${#TASKS[@]} cap=${CAP} poll=${POLL_SECS}s"

while true; do
    all_done=1
    for t in "${TASKS[@]}"; do
        IFS='|' read -r name model dataset metric strategy rays_json out_dir tol extra <<< "$t"
        nl=${NEXT_LINK[$name]}
        if [ "$nl" -le "$LINKS_PER_TASK" ]; then
            all_done=0
        fi
    done
    [ "$all_done" -eq 1 ] && { echo "[$(date)] all chains fully submitted. exiting."; break; }

    n=$(in_flight)
    if [ "$n" -lt "$CAP" ]; then
        for t in "${TASKS[@]}"; do
            IFS='|' read -r name model dataset metric strategy rays_json out_dir tol extra <<< "$t"
            nl=${NEXT_LINK[$name]}
            if [ "$nl" -le "$LINKS_PER_TASK" ]; then
                dep="${LAST_JOBID[$name]:-}"
                jid=$(submit_link "$name" "$model" "$dataset" "$metric" "$strategy" \
                                   "$rays_json" "$out_dir" "$tol" "$extra" "$nl" "$dep")
                if [ -n "$jid" ]; then
                    echo "[$(date)] submitted ${name} link ${nl} -> job ${jid} (in_flight was ${n})"
                    LAST_JOBID["$name"]="$jid"
                    NEXT_LINK["$name"]=$((nl + 1))
                else
                    echo "[$(date)] sbatch FAILED for ${name} link ${nl}, will retry next poll"
                fi
                break   # submit at most one job per poll iteration
            fi
        done
    else
        echo "[$(date)] in_flight=${n} >= cap=${CAP}, waiting..."
    fi
    sleep ${POLL_SECS}
done
