#!/bin/bash

#OOD_DATA=../dataset/sub_1000_openr1.parquet
TRAIN_DATA=../dataset/sub_8000_openr1.parquet

OUTPUT_DIR=./track_results/qwen-math-7b-on-policy/
TEMPLATE=own

# Define arrays for models and their paths
# /jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
MODEL_PATHS=(
  /jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think/
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy_tracking/track_step_30/actor/
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy_tracking/track_step_60/actor/
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy_tracking/track_step_90/actor/
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy_tracking/track_step_120/actor/
)

MODEL_NAMES=(
  Init
  Step_30
  Step_60
  Step_90
  Step_120
)

# Check if arrays have same length
if [ ${#MODEL_PATHS[@]} -ne ${#MODEL_NAMES[@]} ]; then
  echo "Error: MODEL_PATHS and MODEL_NAMES arrays must have the same length"
  exit 1
fi

# Loop through each model
for i in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH=${MODEL_PATHS[$i]}
  MODEL_NAME=${MODEL_NAMES[$i]}
  
  echo "Processing model: $MODEL_NAME"
  echo "Model path: $MODEL_PATH"
  
  CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_acc.py \
    --n 8 \
    --temperature 1.0 \
    --model_path "$MODEL_PATH" \
    --input_file "$TRAIN_DATA" \
    --remove_system True \
    --output_file "$OUTPUT_DIR/sub_8000_openr1_${MODEL_NAME}_acc.parquet" \
    --template "$TEMPLATE"
  
  echo "Completed processing for $MODEL_NAME"
  echo ""
done

echo "All models processed successfully"