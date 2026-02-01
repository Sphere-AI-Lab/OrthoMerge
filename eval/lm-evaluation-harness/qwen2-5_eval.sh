/home/yangsihan/models/Qwen2.5-3B
/home/yangsihan/models/qwen2-5_3b_finetune_tulu-3-code/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_socialiqa/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_commonsense/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_scienceqa/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_tulu-3-algebra/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05

/home/yangsihan/models/qwen2-5_3b_finetune_tulu-3-code/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_socialiqa/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_commonsense/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_scienceqa/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/qwen2-5_3b_finetune_tulu-3-algebra/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05



cd /data/yangsihan/OrthogonalModelMerging/OFT_LLM
conda activate oft_llm
export CUDA_VISIBLE_DEVICES=4
python /data/yangsihan/OrthogonalModelMerging/OFT_LLM/eval_scienceqa.py --merged_dir /home/yangsihan/models/Qwen2.5-3B-Instruct
conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
lm_eval --model hf \
    --tasks social_iqa,commonsense_qa,minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/Qwen2.5-3B-Instruct \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code
conda activate bigcode; cd /data/yangsihan/bigcode-evaluation-harness
accelerate launch  main.py \
  --model /home/yangsihan/models/Qwen2.5-3B-Instruct \
  --max_length_generation 4096 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token


# ==============


/home/yangsihan/models/Qwen2.5-3B--Instruct

conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
export HF_ALLOW_CODE_EVAL=1
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
    --tasks social_iqa \
    --model_args pretrained=/home/yangsihan/models/qwen2-5_3b-Instruct_finetune_socialiqa/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code
conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
export HF_ALLOW_CODE_EVAL=1
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
    --tasks commonsense_qa \
    --model_args pretrained=/home/yangsihan/models/qwen2-5_3b-Instruct_finetune_commonsense/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code
conda activate lm-eval; cd /data/yangsihan/lm-evaluation-harness
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
    --tasks minerva_math500 \
    --model_args pretrained=/home/yangsihan/models/llama3-1_8b_finetune_fast_math/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
    --device cuda:0 \
    --batch_size 8 \
    --confirm_run_unsafe_code



conda activate bigcode; cd /data/yangsihan/bigcode-evaluation-harness
export CUDA_VISIBLE_DEVICES=1
accelerate launch  main.py \
  --model /home/yangsihan/models/qwen2-5_3b-Instruct_finetune_tulu-3-code/oft-finetune_merged_model_oftBS32_lr2p0e-04_ep2_bs64_drop0.05 \
  --max_length_generation 2048 \
  --precision bf16 \
  --tasks humanevalplus \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --use_auth_token


/home/yangsihan/models/llama3-1_8b_finetune_commonsense/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/llama3-1_8b_finetune_magicoder/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/llama3-1_8b_finetune_numinamath/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/llama3-1_8b_finetune_scienceqa/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05
/home/yangsihan/models/llama3-1_8b_finetune_socialiqa/lora-finetune_merged_model_loraR32_lr2p0e-04_ep2_bs64_drop0.05