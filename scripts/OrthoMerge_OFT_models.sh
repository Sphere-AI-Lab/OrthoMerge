#!/bin/bash

set -e 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OrthoMerge
python merge/OrthoMerge_OFT_models.py \
  --language_model_name pathTo/Llama-3.1-8B \
  --adapter_paths \
      pathTo/llama3-1_8b_finetune_magicoder_adapter/ \
      pathTo/llama3-1_8b_finetune_numinamath_adapter/ \
      pathTo/llama3-1_8b_finetune_commonsense_adapter/ \
      pathTo/llama3-1_8b_finetune_socialiqa_adapter/ \
      pathTo/llama3-1_8b_finetune_scienceqa_adapter/ \
  --output_merged_adapter_dir pathTo/OrthoMerge_Llama-3.1-8B \
  --save_merged_model

