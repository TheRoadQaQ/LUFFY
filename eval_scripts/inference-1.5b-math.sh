#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

MODEL_PATH="/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/1.5b_INTERLEAVE/best/actor/"
MODEL_NAME="Math-1.5B-RL-iFT"
TEMPLATE="own"

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own" &

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file "$DATA" \
  --remove_system True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "own" 

#python  /jizhicfs/hymiezhao/ml/busy.py