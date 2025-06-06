#!/bin/bash
#DATA=../dataset/valid.gpqa.parquet
#OUTPUT_DIR=./results/gpqa

DATA=../dataset/valid.mmlu_pro.parquet
OUTPUT_DIR=./results/ood/mmlu

# 定义模型路径、名称、模板的数组
MODEL_PATHS=(
    #"/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_LUFFY_TEST/best/actor/"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE/best/actor/"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy/best/actor/"
    #"/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SFT/epoch2.5/"
    #"/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-Instruct-16-think"
    "/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-16k-think"
    "/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/1.5b_on_policy/best/actor/"
)

MODEL_NAMES=(
    #"Qwen-Math-7B"
    #"LUFFY-7B"
    #"INTERLEAVE-7B"
    #"RL-7B"
    #"SFT-7B"
    #"math-Instruct-7B"
    "math-1.5B"
    "math-1.5B-RL"
)

TEMPLATES=(
    "own"
    "own"
    "own"
    "own"
    "own"
)

# 循环跑每个模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}
    echo "Running $MODEL_NAME ..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE"
done