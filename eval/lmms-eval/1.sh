export CUDA_VISIBLE_DEVICES=2
conda activate lmms-eval; cd /data/yangsihan/lmms-eval
export HF_HOME="~/.cache/huggingface"
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=/home/yangsihan/models/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False \
    --tasks mmsi_bench,erqa,ocrbench,ocrbench_v2 \
    --batch_size 2

conda activate lmms-eval; cd /data/yangsihan/HuatuoGPT-Vision
accelerate launch eval.py --data_path /home/yangsihan/data/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json  --model_path /home/yangsihan/models/Qwen2.5-VL-7B-Instruct