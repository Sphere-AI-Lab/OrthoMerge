/home/yangsihan/models/Qwen2.5-VL-7B-Instruct
/home/yangsihan/models/olmOCR-2-7B-1025-Qwen2.5-VL-7B
/home/yangsihan/models/SenseNova-SI-1.1-Qwen2.5-VL-7B
/home/yangsihan/models/HuatuoGPT-Vision-7B-Qwen2.5VL
# /home/yangsihan/models/MMR1-7B-RL_Qwen2.5-VL-7B

/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/TIES_0-2_1-0
/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/neuron-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean
/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean
# /home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/element-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean
# /home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-Row-Normalization-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval     --model qwen2_5_vl     --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/TIES_0-2_1-0,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2     --tasks charxiv_val_descriptive     --batch_size 2     --output_path ./logs/
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval     --model qwen2_5_vl     --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/neuron-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2     --tasks charxiv_val_descriptive     --batch_size 2     --output_path ./logs/
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval     --model qwen2_5_vl     --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2     --tasks charxiv_val_descriptive     --batch_size 2     --output_path ./logs/

# ,mathvista_test,mathverse_testmini,mathvision_test
# mmsi_bench,blink
# ,dynamath_reasoning,wemath_testmini_reasoning
# ,attn_implementation=flash_attention_2


# mmsi_bench
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="~/.cache/huggingface"
conda activate lmms-eval; cd /data/yangsihan/lmms-eval
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/TIES_0-2_1-0,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmsi_bench,embspatial \
    --batch_size 1

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/neuron-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks embspatial \
    --batch_size 1

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks embspatial \
    --batch_size 1

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/element-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks embspatial \
    --batch_size 1


export CUDA_VISIBLE_DEVICES=4
conda activate lmms-eval; cd /data/yangsihan/HuatuoGPT-Vision
accelerate launch eval.py --data_path /home/yangsihan/data/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json  --model_path /home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-Row-Normalization-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean

chartqa,docvqa_test,textvqa_test,ocrbench_v2都不涨点

,charxiv_val_descriptive
/home/yangsihan/models/Qwen2.5-VL-7B-Instruct

export OPENAI_API_KEY="sk-ph12MU4BJu8711jPN9Siauk7XgdTHtB9d6jxOofS6TuW2xas"
export OPENAI_BASE_URL="https://yinli.one/v1"
export MODEL_VERSION="gpt-4o-mini"
export CUDA_VISIBLE_DEVICES=2
conda activate lmms-eval; cd /data/yangsihan/lmms-eval
export HF_HOME="~/.cache/huggingface"
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks charxiv_val_descriptive \
    --batch_size 2 \
    --verbosity DEBUG


export CUDA_VISIBLE_DEVICES=4
conda activate lmms-eval; cd /data/yangsihan/lmms-eval
export HF_HOME="~/.cache/huggingface"

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval     --model qwen2_5_vl     --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/expert-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2     --tasks mmbench_en,ifeval     --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval     --model qwen2_5_vl     --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/TIES_0-2_1-0,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2     --tasks mmbench_en,ifeval     --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmbench_en,ifeval \
    --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/olmOCR-2-7B-1025-Qwen2.5-VL-7B,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmbench_en,ifeval \
    --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/SenseNova-SI-1.1-Qwen2.5-VL-7B,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmbench_en,ifeval \
    --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/HuatuoGPT-Vision-7B-Qwen2.5VL,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmbench_en,ifeval \
    --batch_size 2
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Merged_Qwen2-5-VL-7B-Instruct/element-conflict-target_TIES_layerwise-compute-target-procrustes_multiGPU_layerwise-calDelta-beforeTIES_theta-mean,max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2 \
    --tasks mmbench_en,ifeval \
    --batch_size 2