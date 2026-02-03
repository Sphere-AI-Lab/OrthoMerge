#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate OrthoMerge
python pathTo/OrthogonalModelMerging/OFT_LLM/eval_scienceqa.py --merged_dir pathTo/OrthoMerge_Llama-3.1-8B/merged_model
conda activate lm-eval; cd pathTo/lm-evaluation-harness
lm_eval --model hf \
    --tasks social_iqa,commonsense_qa,minerva_math500 \
    --model_args pretrained=pathTo/OrthoMerge_Llama-3.1-8B/merged_model \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code
conda activate bigcode; cd pathTo/bigcode-evaluation-harness
accelerate launch  main.py \
  --model pathTo/OrthoMerge_Llama-3.1-8B/merged_model \
  --max_length_generation 4096 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token