#!/bin/bash

#DATA=../dataset/llama_valid.all.parquet
#OUTPUT_DIR=./results

#DATA=../dataset/valid.gpqa.parquet
#OUTPUT_DIR=./results/ood/gpqa

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu


MODEL_PATH="/jizhicfs/hymiezhao/models/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="8B-Llama-instruct"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system False \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE" &

MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/llama_8_SFT/global_step_1071"
MODEL_NAME="8B-Llama-instruct-SFT"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system False \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE"

python /jizhicfs/hymiezhao/ml/busy.py