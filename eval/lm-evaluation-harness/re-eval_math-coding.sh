# /home/yangsihan/models/Qwen3-1.7B
# /home/yangsihan/models/qwen3_1.7b_full_seq_length/magicoder_lora_merged_model_epoch2_lr2e-4_bs64_r32_alpha64_dropout0-05
# /home/yangsihan/models/qwen3_1.7b/merged/NuminaMath_lora_merged_model_epoch2_lr2e-4_bs64_r32_alpha64_dropout0-05

# /home/yangsihan/models/qwen3_1.7b_full_seq_length/magicoder_merged_model_epoch2_lr2e-4_bs64_dropout0-05
# /home/yangsihan/models/qwen3_1.7b/merged/NuminaMath_merged_model_epoch2_lr2e-4_bs64_dropout0-05


# /home/yangsihan/models/qwen3_1.7b_model_merging/TIES_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath_search_hyperparameter/hp_d0.2_l1
# /home/yangsihan/models/qwen3_1.7b_model_merging/TIES_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath
# /home/yangsihan/models/qwen3_1.7b_model_merging/TA_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath

# /home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_Ax_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model
# /home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model
source $(conda info --base)/etc/profile.d/conda.sh

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=4
accelerate launch  main.py \
  --model /home/yangsihan/models/Qwen3-4B \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token
conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=4
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_4b_finetune_magicoder/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/Qwen3-4B \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code



conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_4b_finetune_numinamath/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_full_seq_length/magicoder_merged_model_epoch2_lr2e-4_bs64_dropout0-05 \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_full_seq_length/NuminaMath_merged_model_epoch2_lr2e-4_bs64_dropout0-05 \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_model_merging/TIES_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath_search_hyperparameter/hp_d0.2_l1 \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_model_merging/TIES_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath_search_hyperparameter/hp_d0.2_l1 \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code


conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_model_merging/TIES_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_model_merging/TIES_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_model_merging/TA_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_model_merging/TA_lora_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_Ax_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_Ax_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code

conda activate bigcode-qwen3; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=5
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humaneval \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/qwen3_1.7b_model_merging/OM_avg_Q_oft_commonsenseqa_magicoder_socialiqa_scienceqa_numinamath/merged_model \
    --device cuda:0 \
    --batch_size 1 \
    --confirm_run_unsafe_code