#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

#DATA=../dataset/valid.mmlu_pro.parquet
#OUTPUT_DIR=./results/ood/mmlu

MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7base_INTERLEAVE/best/actor/"
MODEL_NAME="Interleave-7B-Base"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE" &

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE"

#MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7base_on_policy/best/actor/"
#MODEL_NAME="RL-7B-Base"
#TEMPLATE="own"

#"CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
#  --model_path "$MODEL_PATH" \
#  --input_file "$DATA" \
#  --remove_system True \
#  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
#  --template "$TEMPLATE"

#python /jizhicfs/hymiezhao/ml/busy.py