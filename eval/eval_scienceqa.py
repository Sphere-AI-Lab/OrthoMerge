import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import re
import argparse


dataset_name = "derek-thomas/ScienceQA"

parser = argparse.ArgumentParser(description='读取合并模型目录路径')
parser.add_argument('--merged_dir', 
                    type=str,
                    default="OrthoMerge",
                    help='path to model')

args = parser.parse_args()

merged_dir = args.merged_dir

raw_test_dataset = load_dataset(dataset_name, split="test")

def filter_no_image(example):
    img = example.get("image", None)
    if img is None:
        return True
    if isinstance(img, str) and img.strip() == "":
        return True
    return False

test_dataset = raw_test_dataset.filter(filter_no_image)

if "image" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["image"])

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def formatting_prompts_func(example):
    question = example["question"]
    options = example["choices"]  # List[str]
    answer_index = int(example["answer"])
    answer_letter = INDEX_TO_LETTER.get(answer_index, "A")

    choice_lines = []
    for i, opt in enumerate(options):
        label = INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        choice_lines.append(f"{label}. {opt}")
    choices_str = "\n".join(choice_lines)

    text = (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer: {answer_letter}"
    )
    return text


eval_split_name = "test"
eval_dataset = test_dataset  
eval_model = AutoModelForCausalLM.from_pretrained(
    merged_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
eval_tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
eval_tokenizer.pad_token = eval_tokenizer.eos_token
eval_model.eval()


def extract_choice_letter(generated_text):
    m = re.search(r"\b([A-E])\b", generated_text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def build_prompt_from_example(example):
    question = example["question"]
    options = example["choices"]  # List[str]

    choice_lines = []
    for i, opt in enumerate(options):
        label = INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        choice_lines.append(f"{label}. {opt}")
    choices_str = "\n".join(choice_lines)

    prompt = (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer:"
    )
    return prompt


def evaluate_on_scienceqa(model, tokenizer, dataset, max_samples=None):
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        gold_index = int(example["answer"])
        gold_answer = INDEX_TO_LETTER.get(gold_index, "A")

        prompt = build_prompt_from_example(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        pred_choice = extract_choice_letter(generated)

        is_correct = (pred_choice is not None) and (pred_choice == gold_answer)
        if is_correct:
            correct += 1
        total += 1


    acc = correct / total if total > 0 else 0.0
    print(f"Acc: {acc:.4f}")
    return acc


evaluate_on_scienceqa(eval_model, eval_tokenizer, eval_dataset, max_samples=None)

