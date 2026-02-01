#!/usr/bin/env bash

set -euo pipefail

# ====== 必须：显式加载 conda 初始化脚本 ======
# 按照你机器上 CONDA_EXE 的路径，这里是 /data/yangsihan/miniconda3
source "/data/yangsihan/miniconda3/etc/profile.d/conda.sh"

# ====== 可配置路径 ======
ROOT_DIR="/home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_Ax_rescale_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath"
OFT_LLM_DIR="/data/yangsihan/OrthogonalModelMerging/OFT_LLM"
LMEVAL_DIR="/data/yangsihan/lm-evaluation-harness"
GPU_ID="7"

# 遍历所有 coef_xxx 目录
for COEF_DIR in "${ROOT_DIR}"/coef_*; do
    # 确保是目录
    if [[ ! -d "${COEF_DIR}" ]]; then
        continue
    fi

    MERGED_MODEL_DIR="${COEF_DIR}/merged_model"
    if [[ ! -d "${MERGED_MODEL_DIR}" ]]; then
        echo "[WARN] merged_model not found in ${COEF_DIR}, skip."
        continue
    fi

    COEF_NAME="$(basename "${COEF_DIR}")"
    echo
    echo "==============================================================================="
    echo "Processing ${COEF_NAME}  (model: ${MERGED_MODEL_DIR})"
    echo "==============================================================================="

    SCI_LOG="${COEF_DIR}/scienceqa_eval.log"
    LM_LOG="${COEF_DIR}/lm_eval.log"

    # =========================
    # 1) 在 oft_llm 环境中跑 ScienceQA
    # =========================
    echo
    echo "[INFO] Running ScienceQA eval for ${COEF_NAME} ..."
    (
        cd "${OFT_LLM_DIR}"
        conda activate oft_llm
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${OFT_LLM_DIR}/eval_scienceqa.py" \
            --merged_dir "${MERGED_MODEL_DIR}"
    ) 2>&1 | tee "${SCI_LOG}"

    # =========================
    # 2) 在 lm-eval 环境中跑 lm_eval
    # =========================
    echo
    echo "[INFO] Running lm-eval for ${COEF_NAME} ..."
    (
        cd "${LMEVAL_DIR}"
        conda activate lm-eval
        export HF_ALLOW_CODE_EVAL=1
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        lm_eval --model hf \
            --tasks social_iqa,commonsense_qa,mbpp,mathqa \
            --model_args "pretrained=${MERGED_MODEL_DIR}" \
            --device "cuda:0" \
            --batch_size 1 \
            --confirm_run_unsafe_code
    ) 2>&1 | tee "${LM_LOG}"

    echo
    echo "[INFO] Finished ${COEF_NAME}. Logs saved to:"
    echo "  - ${SCI_LOG}"
    echo "  - ${LM_LOG}"
    echo "==============================================================================="
done

echo
echo "All coefficients processed."