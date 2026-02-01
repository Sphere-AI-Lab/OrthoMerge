import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
import json
import tempfile
import shutil

from typing import List, Literal, Optional
import time



def merge_cayley_Q_list(
    weights_list: List[torch.Tensor],
) -> torch.Tensor:
    print("product C")
    assert len(weights_list) > 0, "weights_list none"
    n_task = len(weights_list)

    Q_stack = torch.stack(weights_list, dim=0)

    merged_sum = Q_stack.sum(dim=0)

    # sum_{i=0}^{N-1} |delta_i|_F
    N = Q_stack.shape[0]
    norms = torch.norm(Q_stack.view(N, -1), p='fro', dim=1) 
    sum_of_norms = norms.sum()

    # |sum_{i=0}^{N-1} delta_i|_F
    norm_of_sum = torch.norm(merged_sum, p='fro')

    c = sum_of_norms / (norm_of_sum)

    merged = (1/n_task) * c * merged_sum

    merged = 0.5 * (merged - merged.transpose(-1, -2))
    
    return merged


def oft_params_to_skew_matrix(oft_params: torch.Tensor, block_size: int = 32) -> torch.Tensor:

    num_blocks, num_params = oft_params.shape
    expected = block_size * (block_size - 1) // 2
    if num_params != expected:
        raise ValueError(
            f"num_params_per_block={num_params}  block_size={block_size} "
        )

    indices = torch.triu_indices(block_size, block_size, offset=1, device=oft_params.device)
    rows, cols = indices[0], indices[1]
    S = torch.zeros(num_blocks, block_size, block_size,
                    dtype=oft_params.dtype, device=oft_params.device)

    S[:, rows, cols] = oft_params
    S = S - S.transpose(-2, -1)

    return S


def skew_matrix_to_oft_params(S: torch.Tensor) -> torch.Tensor:

    num_blocks, block_size, _ = S.shape


    indices = torch.triu_indices(block_size, block_size, offset=1, device=S.device)
    rows, cols = indices[0], indices[1]
    oft_params = S[:, rows, cols]  # (num_blocks, num_params_per_block)

    return oft_params

cache_dir = None  


parser = argparse.ArgumentParser("Interface for merging LLMs with multiple OFT adapters (no evaluation)")
parser.add_argument(
    "--language_model_name",
    type=str,
    required=True,
    help="Base LLM name or path, e.g., 'meta-llama/Llama-2-7b-hf' or local path"
)
parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
parser.add_argument(
    "--adapter_paths",
    type=str,
    nargs='+',
    required=True,
    help="Paths to the saved OFT adapter directories (can provide multiple)"
)
parser.add_argument(
    "--output_merged_adapter_dir",
    type=str,
    default="./merged_oft_adapter",
    help="Where to save merged adapter and/or merged base model"
)
parser.add_argument(
    "--save_merged_model",
    action="store_true",
    help="If set, will save base model with merged weights (merge_and_unload) to output_merged_adapter_dir/model"
)
parser.add_argument(
    "--just_merge_adapter",
    action="store_true",
    help="If set, only merge adapter weights and save a merged adapter folder, without loading base model"
)


args = parser.parse_args()
if torch.cuda.is_available() and args.gpu >= 0:
    args.device = f"cuda:{args.gpu}"
else:
    args.device = "cpu"


def merge_oft_adapter_weights(adapter_paths):
    print(f"\nMerging {len(adapter_paths)} OFT adapters...")

    all_weights = []
    for adapter_path in adapter_paths:
        model_path = os.path.join(adapter_path, 'adapter_model.safetensors')
        if not os.path.exists(model_path):
            model_path = os.path.join(adapter_path, 'adapter_model.bin')
            if os.path.exists(model_path):
                weights = torch.load(model_path, map_location='cpu')
            else:
                print(f"Warning: No adapter weights found at {adapter_path}, skipping...")
                continue
        else:
            weights = load_file(model_path)

        all_weights.append(weights)
        print(f"  Loaded: {adapter_path}")

    if not all_weights:
        raise ValueError("No valid adapter weights found!")

    merged_weights = {}
    first_weights = all_weights[0]

    for key in first_weights.keys():
        print(f"  Processing key: {key}")

        if 'oft_r' in key or ('oft_' in key.lower() and 'classifier' not in key.lower()):
            weights_list = []
            for weights in all_weights:
                if key in weights:
                    w = weights[key]
                    w = oft_params_to_skew_matrix(w)
                    weights_list.append(w)

            if weights_list:
                avg_weight = merge_cayley_Q_list(weights_list)

                avg_weight = skew_matrix_to_oft_params(avg_weight)
                merged_weights[key] = avg_weight
                print(f"    Merged shape: {avg_weight.shape}")
            else:
                print(f"    Warning: No weights found for key {key}")

        else:
            weights_list = [weights[key] for weights in all_weights if key in weights]
            if len(weights_list) > 1:
                avg_weight = torch.stack(weights_list).mean(dim=0)
                merged_weights[key] = avg_weight
                print(f"    Averaged non-OFT weight, shape: {avg_weight.shape}")
            else:
                merged_weights[key] = first_weights[key].clone()
                print(f"    Copied from first adapter, shape: {merged_weights[key].shape}")

    print(f"  Merged {len(merged_weights)} weight tensors")
    return merged_weights


def create_merged_adapter_with_oft_for_llm(base_model_name, adapter_paths,
                                           output_merged_adapter_dir,
                                           save_merged_model: bool = False,
                                           device: str = "cpu"):
    print(f"\n{'=' * 80}")
    print("Creating merged OFT adapter for LLM")
    print(f"Base model: {base_model_name}")
    print(f"{'=' * 80}")

    config_path = os.path.join(adapter_paths[0], 'adapter_config.json')

    merged_weights = merge_oft_adapter_weights(adapter_paths)

    os.makedirs(output_merged_adapter_dir, exist_ok=True)
    merged_adapter_path = os.path.join(output_merged_adapter_dir, "merged_adapter")
    os.makedirs(merged_adapter_path, exist_ok=True)

    merged_weights_path = os.path.join(merged_adapter_path, 'adapter_model.safetensors')
    save_file(merged_weights, merged_weights_path)
    shutil.copy(config_path, os.path.join(merged_adapter_path, 'adapter_config.json'))

    print(f"  Saved merged adapter to: {merged_adapter_path}")

    if not save_merged_model:
        print("  [INFO] save_merged_model = False, stop after saving merged adapter.")
        print(f"{'=' * 80}\n")
        return None, merged_adapter_path

    print("\nLoading base LLM (causal LM)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=(os.path.join(cache_dir, base_model_name) if cache_dir else base_model_name),
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,  
            device_map=None  
        )
    except Exception as e:
        print(f"  Failed to load from cache_dir, fallback to {base_model_name} directly. Error: {e}")
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map=None
        )

    base_model.to(device)
    print(f"  Base model loaded on {device}")

    print("  Loading merged adapter with PeftModel and merging into base model...")
    peft_model = PeftModel.from_pretrained(base_model, merged_adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.to(device)

    print("  Successfully merged encoder weights into LLM (merge_and_unload finished).")

    model_save_dir = os.path.join(output_merged_adapter_dir, "merged_model")
    os.makedirs(model_save_dir, exist_ok=True)
    merged_model.save_pretrained(model_save_dir)
    print(f"  Saved merged base model to: {model_save_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    tokenizer.save_pretrained(model_save_dir)


    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{'=' * 80}\n")
    return merged_model, merged_adapter_path


if __name__ == "__main__":
    print("=" * 80)
    print(f"Processing {len(args.adapter_paths)} adapters:")
    for i, path in enumerate(args.adapter_paths, 1):
        print(f"  {i}. {path}")
    print(f"\nBase Model: {args.language_model_name}")
    print(f"Device: {args.device}")
    print(f"Output Dir: {args.output_merged_adapter_dir}")
    print("=" * 80)
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=(
                os.path.join(cache_dir, args.language_model_name) if cache_dir else args.language_model_name
            ),
            cache_dir=cache_dir
        )
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Failed to load tokenizer, error: {e}")
        tokenizer = None

    merged_model, merged_adapter_path = create_merged_adapter_with_oft_for_llm(
        base_model_name=args.language_model_name,
        adapter_paths=args.adapter_paths,
        output_merged_adapter_dir=args.output_merged_adapter_dir,
        save_merged_model=not args.just_merge_adapter,
        device=args.device
    )

    sys.exit()



