#!/bin/bash

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

# 定义模型路径、名称、模板的数组
MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE/best/actor/"
MODEL_NAME="7B-interleave"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE"

DATA=../dataset/valid.gpqa.parquet
OUTPUT_DIR=./results/ood/gpqa

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE"

python /jizhicfs/hymiezhao/ml/busy.py