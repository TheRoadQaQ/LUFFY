#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

#DATA=../dataset/valid.mmlu_pro.parquet
#OUTPUT_DIR=./results/ood/mmlu

MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7base_SFT/global_step_1071"
MODEL_NAME="SFT-epoch3-7B-Base"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE"

#python /jizhicfs/hymiezhao/ml/busy.py