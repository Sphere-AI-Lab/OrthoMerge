import os
import gc
import copy
import random
import math
from collections import OrderedDict
from typing import List, Dict, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import torch
from torch import nn, Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

BASE_MODEL_PATH = "pathTo/models/Llama-3.2-3B"

PAIR_PATHS = [
    ("pathTo/models/Llama-3.2-3B", "pathTo/models/MergeBench/Llama-3.2-3B_coding"),
    ("pathTo/models/Llama-3.2-3B", "pathTo/models/MergeBench/Llama-3.2-3B_instruction"),
    ("pathTo/models/Llama-3.2-3B", "pathTo/models/MergeBench/Llama-3.2-3B_math"),
    ("pathTo/models/Llama-3.2-3B", "pathTo/models/MergeBench/Llama-3.2-3B_multilingual"),
    ("pathTo/models/Llama-3.2-3B", "pathTo/models/MergeBench/Llama-3.2-3B_safety"),
]

OUTPUT_PATH = "pathTo/models/Llama-3.2-3B_OrthoMerge-G-TIES"
CAYLEY_CACHE_DIR = "./cache_Llama-3.2-3B_G"



USE_EXISTING_CACHE = True

RESET_THRESH = 0.2
REMOVE_KEYS: List[str] = []

TIES_CHUNK_SIZE = 50_000_000 

AVAILABLE_GPUS = [i for i in range(torch.cuda.device_count())]

def seed_torch(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


_TOKENIZER_CACHE = {}

def get_tokenizer(model_path: str):
    if model_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _TOKENIZER_CACHE[model_path]

def remap_embedding_to_base_vocab(base_vocab, src_vocab, src_embed):
    vocab_size = len(base_vocab)
    embed_size = src_embed.shape[1]
    device = src_embed.device
    dtype = src_embed.dtype
    
    new_embed = torch.zeros((vocab_size, embed_size), dtype=dtype, device=device)
    
    for token, base_id in base_vocab.items():
        if token in src_vocab:
            src_id = src_vocab[token]
            if src_id < src_embed.shape[0]:
                new_embed[base_id] = src_embed[src_id]
         
    return new_embed

class VectorIndexManager:
    def __init__(self, state_dict: Dict[str, Tensor], remove_keys: List[str] = []):
        self.indices = OrderedDict()
        self.total_params = 0
        self.shapes = {}
        
        keys = sorted([k for k in state_dict.keys() if k not in remove_keys])
        
        current_idx = 0
        for key in keys:
            tensor = state_dict[key]
            numel = tensor.numel()
            self.indices[key] = (current_idx, current_idx + numel)
            self.shapes[key] = tensor.shape
            current_idx += numel
            self.total_params += numel
            
    def get_slice(self, vector: Tensor, key: str) -> Tensor:
        if key not in self.indices:
            return None
        start, end = self.indices[key]
        flat_slice = vector[start:end] 
        return flat_slice.view(self.shapes[key])

    def get_slice_from_distributed(self, distributed_vectors: List[Tensor], key: str, device: str) -> List[Tensor]:
        start, end = self.indices[key]
        shape = self.shapes[key]
        results = []
        for vec in distributed_vectors:
            slice_data = vec[start:end].view(shape).to(device, non_blocking=True)
            results.append(slice_data)
        return results


def orthogonal_procrustes_torch_right(W1: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
    # Find R such that W0 @ R \approx W1
    # W1: Target (Expert) [out, in]
    # W0: Source (Base)   [out, in]
    A = torch.matmul(W0.t(), W1)
    try:
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    except:
        U, _, Vh = torch.linalg.svd(A.cpu(), full_matrices=False)
        U = U.to(A.device)
        Vh = Vh.to(A.device)
    return torch.matmul(U, Vh)

def cayley_to_skew(R: torch.Tensor) -> torch.Tensor:
    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    R_f = R.float()
    I_f = I.float()
    A = torch.linalg.solve(R_f + I_f, R_f - I_f)
    A = 0.5 * (A - A.transpose(-1, -2))
    return A

def cayley_from_skew(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A_f = A.float()
    I_f = I.float()
    R = torch.linalg.solve(I_f - A_f, I_f + A_f)
    return R


def merge_cayley_Q_list(
    weights_list: List[torch.Tensor],
    theta_agg: str = "mean",
    direction_weight: str = "theta",
) -> torch.Tensor:
    
    base_shape = weights_list[0].shape
    Q_stack = torch.stack(weights_list, dim=0)  # [T, d, d]
    T = Q_stack.shape[0]

    Q_flat = Q_stack.reshape(T, -1)  # [T, N]
    N = Q_flat.size(1)

    theta = torch.linalg.vector_norm(Q_flat, dim=1)  # [T]
    
    theta_clamped = torch.clamp(theta, min=1e-8)
    u = Q_flat / theta_clamped.unsqueeze(1)  # [T, N]

    # 2. 确定方向聚合的权重
    if direction_weight == "theta":
        w = theta.clone()
    elif direction_weight == "uniform":
        w = torch.ones_like(theta)
    else:
        raise ValueError(f"Unknown direction_weight: {direction_weight}")

    u_weighted = u * w.unsqueeze(1)  # [T, N]
    u_sum = u_weighted.sum(dim=0)    # [N]

    u_sum_norm = torch.linalg.vector_norm(u_sum)
    
    if u_sum_norm < 1e-8:
        return torch.zeros(base_shape, device=Q_flat.device, dtype=Q_flat.dtype)
    
    u_merge = u_sum / u_sum_norm  

    if theta_agg == "mean":
        theta_merge = theta.mean()
    elif theta_agg == "median":
        theta_merge = theta.median()
    elif theta_agg == "max":
        theta_merge = theta.max()
    else:
        raise ValueError(f"Unknown theta_agg: {theta_agg}")

    merged_flat = u_merge * theta_merge    # [N]
    merged = merged_flat.reshape(base_shape)

    return merged

def load_and_flatten_base(model_path: str) -> Tuple[Tensor, VectorIndexManager]:
    print(f"[Init] Loading Base Model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.bfloat16)
    sd = model.state_dict()
    index_mgr = VectorIndexManager(sd, REMOVE_KEYS)
    sorted_keys = sorted([k for k in sd.keys() if k not in REMOVE_KEYS])
    flat_base = nn.utils.parameters_to_vector([sd[k].reshape(-1) for k in sorted_keys])
    del model, sd
    gc.collect()
    return flat_base, index_mgr

def layerwise_procrustes_merge_with_conflict_targets(
    base_vec_cpu: Tensor,
    index_mgr: VectorIndexManager,
    pairs: List[Tuple[str, str]],
    cache_dir: str,
    run_mode: Literal["GENERATE_CACHE", "READ_CACHE"] = "GENERATE_CACHE",
) -> Tensor:
    print(f"[Layerwise Ortho] Mode = {run_mode}")
    os.makedirs(cache_dir, exist_ok=True)

    updated_base_vec = base_vec_cpu.clone()
    base_tok = get_tokenizer(BASE_MODEL_PATH)
    base_vocab = base_tok.get_vocab()
    compute_device = f"cuda:{AVAILABLE_GPUS[0]}"

    print("  Loading all experts to CPU (warning: high RAM)...")
    loaded_experts = []
    for base_p, expert_p in pairs:
        model = AutoModelForCausalLM.from_pretrained(
            expert_p, device_map="cpu", torch_dtype=torch.bfloat16
        )
        sd = model.state_dict()
        del model

        expert_tok = get_tokenizer(expert_p)
        expert_vocab = expert_tok.get_vocab()
        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            if key in sd:
                w_exp = sd[key]
                if w_exp.shape != index_mgr.shapes[key]:
                    sd[key] = remap_embedding_to_base_vocab(
                        base_vocab, expert_vocab, w_exp
                    )
        loaded_experts.append(sd)

    target_layers = []
    for key, shape in index_mgr.shapes.items():
        if len(shape) == 2 and "weight" in key and "layernorm" not in key:
            target_layers.append(key)

    print(f"  Found {len(target_layers)} layers to process.")

    CONFLICT_THRESHOLD = 0.0

    for layer_name in tqdm(target_layers, desc="Layerwise Ortho+Targets"):
        start, end = index_mgr.indices[layer_name]
        shape = index_mgr.shapes[layer_name]
        safe_name = layer_name.replace(".", "_")
        cache_file = os.path.join(cache_dir, f"{safe_name}_A_list.pt")

        base_slice = base_vec_cpu[start:end].view(shape).to(compute_device)

        A_list = None
        if run_mode == "READ_CACHE" and os.path.exists(cache_file):
            try:
                A_list = torch.load(cache_file, map_location=compute_device)
            except Exception as e:
                print(f"  [WARN] Failed to load cache {cache_file}: {e}")

        if run_mode == "GENERATE_CACHE" and os.path.exists(cache_file):
            try:
                A_list = torch.load(cache_file, map_location=compute_device)
            except Exception:
                A_list = None  

        if A_list is None:
            target_Ws = []
            for exp_sd in loaded_experts:
                if layer_name in exp_sd:
                    w_exp = exp_sd[layer_name].to(compute_device)
                else:
                    w_exp = base_slice.clone()
                target_Ws.append(w_exp)

            base_w = base_slice.float()
            R_list = []
            for target_w in target_Ws:
                R = orthogonal_procrustes_torch_right(target_w.float(), base_w)
                R_list.append(R)

            A_list = [cayley_to_skew(R) for R in R_list]
            torch.save([A.cpu() for A in A_list], cache_file)

            del stack_deltas, mean_delta, target_Ws, R_list, base_w
            torch.cuda.empty_cache()

        if A_list is not None:
            A_merged = merge_cayley_Q_list(A_list)
            R_merged = cayley_from_skew(A_merged)

            base_w = base_slice.float()
            W_merged = base_w @ R_merged

            updated_base_vec[start:end] = (
                W_merged.cpu().to(updated_base_vec.dtype).view(-1)
            )

            del A_list, A_merged, R_merged, base_w, W_merged

        del base_slice
        torch.cuda.empty_cache()

    del loaded_experts
    gc.collect()

    return updated_base_vec

def layerwise_compute_deltas_with_specific_rotations(
    base_vec_cpu: Tensor,
    index_mgr: VectorIndexManager,
    pairs: List[Tuple[str, str]],
    cache_dir: str,
    desc: str = "Computing Refined Deltas"
) -> List[Tensor]:
    print(f"[{desc}] Calculating Expert - (Base * R_i) [layerwise]...")
    distributed_deltas: List[Tensor] = []
    
    base_tok = get_tokenizer(BASE_MODEL_PATH)
    base_vocab = base_tok.get_vocab()

    sorted_keys = sorted([k for k in index_mgr.shapes.keys() if k not in REMOVE_KEYS])

    for task_idx, (base_p, expert_p) in enumerate(pairs):
        target_gpu = AVAILABLE_GPUS[task_idx % len(AVAILABLE_GPUS)]
        device = f"cuda:{target_gpu}"
        print(f"  - Task {task_idx+1}/{len(pairs)}: {expert_p} -> {device} (layerwise, with specific rotation)")
        
        expert_model = AutoModelForCausalLM.from_pretrained(
            expert_p,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )
        expert_sd = expert_model.state_dict()
        del expert_model
        
        expert_tok = get_tokenizer(expert_p)
        expert_vocab = expert_tok.get_vocab()
        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            if key in expert_sd:
                w_exp = expert_sd[key]
                w_base_shape = index_mgr.shapes[key]
                if w_exp.shape != w_base_shape:
                    w_aligned = remap_embedding_to_base_vocab(
                        base_vocab, expert_vocab, w_exp
                    )
                    expert_sd[key] = w_aligned

        flat_delta_cpu = torch.zeros_like(base_vec_cpu, dtype=base_vec_cpu.dtype, device="cpu")

        for layer_name in sorted_keys:
            if layer_name not in expert_sd:
                continue

            start, end = index_mgr.indices[layer_name]
            shape = index_mgr.shapes[layer_name]

            safe_name = layer_name.replace(".", "_")
            cache_file = os.path.join(cache_dir, f"{safe_name}_A_list.pt")

            R_task = None
            if os.path.exists(cache_file):
                try:
                    A_list = torch.load(cache_file, map_location="cpu")
                    if task_idx < len(A_list) and A_list[task_idx] is not None:
                        A_task = A_list[task_idx].to(device=device, dtype=torch.float32, non_blocking=True)
                        R_task = cayley_from_skew(A_task)  # [in, in]
                        del A_task
                    del A_list
                except Exception as e:
                    print(f"    Warning: Failed to load/transform rotation for layer {layer_name}, task {task_idx}: {e}")
                    R_task = None

            base_layer = index_mgr.get_slice(base_vec_cpu, layer_name).to(
                device=device,
                non_blocking=True,
            )  # [out, in]

            expert_layer = expert_sd[layer_name].to(
                device=device,
                non_blocking=True,
            )  # [out, in]

            if R_task is not None:
                # base_layer @ R_task:  [out, in] @ [in, in] -> [out, in]
                rotated_base = torch.matmul(base_layer, R_task.to(base_layer.dtype))
            else:
                rotated_base = base_layer

            # delta_layer: expert_layer - rotated_base
            delta_layer = expert_layer - rotated_base 

            flat_delta_cpu[start:end] = (
                delta_layer.detach().cpu().to(dtype=base_vec_cpu.dtype).view(-1)
            )

            del base_layer, expert_layer, rotated_base, delta_layer, R_task
            torch.cuda.empty_cache()

        delta_gpu = flat_delta_cpu.to(device=device, non_blocking=True)

        del flat_delta_cpu, expert_sd
        torch.cuda.empty_cache()
        gc.collect()

        distributed_deltas.append(delta_gpu)

    return distributed_deltas

def distributed_ties_with_full_discarded(
    distributed_deltas: List[Tensor], 
    reset_thresh: float,
    chunk_size: int = TIES_CHUNK_SIZE
) -> Tuple[Tensor, List[Tensor]]:
    """
    执行 TIES Merge。
    """
    print(f"[TIES] Running Distributed TIES (Top-k: {reset_thresh})...")
    
    num_tasks = len(distributed_deltas)
    total_params = distributed_deltas[0].numel()
    
    # --- Step 1: Local Top-k ---
    print(f"  Phase 1: Local Top-k & Offloading discarded...")
    kept_list_gpu = []
    
    for i, delta in enumerate(distributed_deltas):
        k = int(delta.numel() * (1 - reset_thresh))
        kth_val = delta.abs().kthvalue(k).values
        mask = delta.abs() >= kth_val
        
        kept = delta * mask
        
        kept_list_gpu.append(kept)
        
        distributed_deltas[i] = None 
        del delta
        torch.cuda.empty_cache()

    # --- Step 2: Global Merge ---
    print(f"  Phase 2: Chunked Global Merge (Chunk: {chunk_size})...")
    
    final_merged_delta = torch.zeros(total_params, dtype=kept_list_gpu[0].dtype, device="cpu")
    main_device = kept_list_gpu[0].device
    
    for start_idx in tqdm(range(0, total_params, chunk_size), desc="TIES Merging"):
        end_idx = min(start_idx + chunk_size, total_params)
        
        chunk_sum = torch.zeros(end_idx - start_idx, device=main_device)
        task_chunks = []
        
        for task_idx in range(num_tasks):
            c = kept_list_gpu[task_idx][start_idx:end_idx].to(main_device, non_blocking=True)
            task_chunks.append(c)
            chunk_sum += c
        
        final_signs = torch.sign(chunk_sum)
        if (final_signs == 0).any():
            final_signs[final_signs == 0] = 1.0
            
        chunk_numerator = torch.zeros_like(chunk_sum)
        chunk_denominator = torch.zeros_like(chunk_sum)
        
        for task_idx in range(num_tasks):
            kept_chunk = task_chunks[task_idx]
            rows_to_keep = torch.where(final_signs > 0, kept_chunk > 0, kept_chunk < 0).float()
            kept_final = kept_chunk * rows_to_keep
            
            chunk_numerator += kept_final
            chunk_denominator += (kept_final != 0).float()
            
        merged_chunk = chunk_numerator / torch.clamp(chunk_denominator, min=1)
        final_merged_delta[start_idx:end_idx] = merged_chunk.cpu()
        
        del chunk_sum, task_chunks, final_signs, chunk_numerator, chunk_denominator, merged_chunk
        
    print("  Moving merged result to GPU 0...")
    final_merged_delta_gpu = final_merged_delta.to(main_device)
    
    del kept_list_gpu
    torch.cuda.empty_cache()
    
    return final_merged_delta_gpu

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    seed_torch(42)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print(f"[INFO] Using GPUs: {AVAILABLE_GPUS}")
    
    # 1. Load Original Base & Build Index (CPU)
    base_vec_cpu, index_mgr = load_and_flatten_base(BASE_MODEL_PATH)
    
    should_reuse = False
    if USE_EXISTING_CACHE and os.path.exists(CAYLEY_CACHE_DIR):
        if len(os.listdir(CAYLEY_CACHE_DIR)) > 0:
            should_reuse = True
    
    distributed_targets = None
    run_mode = "GENERATE_CACHE"
    # ================
    print(f"\n=== Layerwise Conflict Targets + Procrustes (Mode = {run_mode}) ===")

    ortho_base_vec_cpu = layerwise_procrustes_merge_with_conflict_targets(
        base_vec_cpu,
        index_mgr,
        PAIR_PATHS,
        CAYLEY_CACHE_DIR,
        run_mode="GENERATE_CACHE" if not should_reuse else "READ_CACHE"
    )
    # ================ 
    if distributed_targets is not None:
        del distributed_targets
    torch.cuda.empty_cache()
    gc.collect()
    
    
    distributed_deltas_r2 = layerwise_compute_deltas_with_specific_rotations(
        base_vec_cpu, 
        index_mgr, 
        PAIR_PATHS, 
        CAYLEY_CACHE_DIR,
        desc="Round 2 Deltas (Expert - Procrustes_Model)"
    )
    
    final_merged_delta_gpu = distributed_ties_with_full_discarded(
        distributed_deltas_r2, RESET_THRESH
    )
    
    del distributed_deltas_r2
    torch.cuda.empty_cache()
    
    # ============================================================
    # Final Addition & Save
    # ============================================================
    print(f"\n[Final] Combining Ortho Base + Scaling * TIES Delta...")
    
    final_vec = ortho_base_vec_cpu + 1.0 * final_merged_delta_gpu.cpu()
    
    del ortho_base_vec_cpu, final_merged_delta_gpu
    gc.collect()
    
    print(f"[Save] Saving to {OUTPUT_PATH}...")
    
    # Reload structure
    print("  Reloading base model structure...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16)
    state_dict = model.state_dict()
    with torch.no_grad():
        for key, (start, end) in index_mgr.indices.items():
            if key in state_dict:
                shape = state_dict[key].shape
                idx_shape = index_mgr.shapes[key]
                if shape != idx_shape:
                    new_tensor = torch.zeros(idx_shape, dtype=state_dict[key].dtype, device="cpu")
                    state_dict[key] = new_tensor
                state_dict[key].copy_(final_vec[start:end].view(idx_shape))
    
    model.load_state_dict(state_dict)
    model.save_pretrained(OUTPUT_PATH)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tok.save_pretrained(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()

