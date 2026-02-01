#!/bin/bash

set -e

BASE_OUTPUT_DIR="/home/yangsihan/models/MergeBench_merged/Llama-3.2-3B_IsoC"

LOG_ROOT="/data/yangsihan/lm-evaluation-harness/eval_all_hp_Llama-3.2-3B_IsoC_logs"
mkdir -p "${LOG_ROOT}"

for MODEL_DIR in "${BASE_OUTPUT_DIR}"/IsoC_SF*; do
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
    bash /data/yangsihan/lm-evaluation-harness/mergebench_eval_all.sh "${MODEL_DIR}" 2 "/data/yangsihan/lm-evaluation-harness/results/Llama-3.2-3B_IsoC/${MODEL_NAME}"
  } 2>&1 | tee -a "${LOG_FILE}"

done