#!/bin/bash
#DATA=../dataset/valid.gpqa.parquet
#OUTPUT_DIR=./results/gpqa

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

# 定义模型路径、名称、模板的数组
MODEL_PATHS=(
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_all_buffer/best/actor/"
    "/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_random_buffer/best/actor/"
    "/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_random2_buffer/best/actor/"
)

MODEL_NAMES=(
    #"Interleave-7B-all"
    "Interleave-7B-random"
    "Interleave-7B-uniform"
)

TEMPLATES=(
    #"own"
    "own"
    "own"
)

# 循环跑每个模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}
    echo "Running $MODEL_NAME ..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE"
done