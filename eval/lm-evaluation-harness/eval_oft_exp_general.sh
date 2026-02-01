#!/bin/bash

MODEL=$1
GPU_ID=$2
OUTPUT_PATH=$3

echo $MODEL
echo $GPU_ID
echo $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=$GPU_ID

source $(conda info --base)/etc/profile.d/conda.sh
mkdir -p $OUTPUT_PATH

conda activate lm-eval
cd /data/yangsihan/lm-evaluation-harness

lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks mmlu,agieval \
    --device cuda:$GPU_ID \
    --batch_size 8 \
    --output_path $OUTPUT_PATH

# truthfulqa,acp_bench,agieval,medqa_4options,medmcqa,crows_pairs_english,discrim_eval_explicit,discrim_eval_implicit


