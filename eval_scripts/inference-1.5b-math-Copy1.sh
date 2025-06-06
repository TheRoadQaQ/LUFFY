#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

MODEL_PATH="/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-16k-think"
MODEL_NAME="Math-1.5B"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=4 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own" &

#####

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

MODEL_PATH="/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-16k-think"
MODEL_NAME="Math-1.5B"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=5 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own" &

####

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

MODEL_PATH="/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Math-1.5B-Instruct"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=6 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own" &

####

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

MODEL_PATH="/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Math-1.5B-Instruct"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=7 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own"


CUDA_VISIBLE_DEVICES=4,5,6,7 python  /jizhicfs/hymiezhao/ml/busy.py