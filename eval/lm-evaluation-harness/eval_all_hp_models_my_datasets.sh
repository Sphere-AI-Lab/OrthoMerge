#!/bin/bash

set -e

BASE_OUTPUT_DIR="/home/yangsihan/models/qwen3_1.7b_model_merging/TIES_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath_search_hyperparameter"

LOG_ROOT="/data/yangsihan/lm-evaluation-harness/eval_all_hp_ties_oft_qwen3-1.7b_logs"
mkdir -p "${LOG_ROOT}"

for MODEL_DIR in "${BASE_OUTPUT_DIR}"/hp_d*_l*; do
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "Skip non-directory: ${MODEL_DIR}"
    continue
  fi

  MODEL_NAME=$(basename "${MODEL_DIR}")  # 比如 hp_d0.1_l0.3
  LOG_FILE="${LOG_ROOT}/${MODEL_NAME}.log"

  echo "=====================================================" | tee "${LOG_FILE}"
  echo "Evaluating model: ${MODEL_DIR}" | tee -a "${LOG_FILE}"
  echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  echo "=====================================================" | tee -a "${LOG_FILE}"

  {
    #######################################
    # 2. 评测 ScienceQA（OFT_LLM 环境）
    #######################################
    source "$(conda info --base)/etc/profile.d/conda.sh"
    cd /data/yangsihan/OrthogonalModelMerging/OFT_LLM || exit 1
    conda activate oft_llm

    echo "[ScienceQA] Using merged_dir = ${MODEL_DIR}"
    CUDA_VISIBLE_DEVICES=1 python /data/yangsihan/OrthogonalModelMerging/OFT_LLM/eval_scienceqa.py \
      --merged_dir "${MODEL_DIR}"

    #######################################
    # 3. 评测 lm-eval-harness（多个任务）
    #######################################
    source "$(conda info --base)/etc/profile.d/conda.sh"
    cd /data/yangsihan/lm-evaluation-harness || exit 1
    conda activate lm-eval

    export HF_ALLOW_CODE_EVAL=1

    echo "[lm-eval] Using pretrained = ${MODEL_DIR}"
    CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
      --tasks mathqa,social_iqa,mbpp,commonsense_qa \
      --model_args "pretrained=${MODEL_DIR}" \
      --device cuda:0 \
      --batch_size 1 \
      --confirm_run_unsafe_code

    echo "Finished evaluation for model: ${MODEL_DIR}"
    echo
  } 2>&1 | tee -a "${LOG_FILE}"

done