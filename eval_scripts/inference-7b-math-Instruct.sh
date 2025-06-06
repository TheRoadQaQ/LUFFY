#!/bin/bash

DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

DATA=../dataset/valid.gpqa.parquet
OUTPUT_DIR=./results/ood/gpqa

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

# 定义模型路径、名称、模板的数组
MODEL_PATHS=(
    "/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-Instruct-16-think"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7base_on_policy/best/actor/"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7base_INTERLEAVE/best/actor/"
)
MODEL_NAMES=(
    "7B-Math_instruct"
    #"RL-7B-base"
    #"INTERLEAVE-7B-base"
)
TEMPLATES=(
    "own"
    #"own"
    #"own"
)

# 遍历所有模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}

    echo "Running inference for $MODEL_NAME ..."

    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE" > "$OUTPUT_DIR/$MODEL_NAME.log"
done

python /jizhicfs/hymiezhao/ml/busy.py