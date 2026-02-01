import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src = "/home/yangsihan/models/gemma-2-9b"
dst = "/home/yangsihan/models/gemma-2-9b-bf16"

print(f"Loading model from {src} ...")
model = AutoModelForCausalLM.from_pretrained(
    src,
    torch_dtype=torch.float32,   # 原始是 FP32
    device_map="cpu",
)

# 转成 bfloat16
model = model.to(torch.bfloat16)

print(f"Saving bf16 model to {dst} ...")
model.save_pretrained(dst)

# tokenizer 原样拷贝
tokenizer = AutoTokenizer.from_pretrained(src)
tokenizer.save_pretrained(dst)

print("Done.")

'''
python /data/yangsihan/lm-evaluation-harness/convert_to_bf16.py
'''