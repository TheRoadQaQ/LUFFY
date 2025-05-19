#!/bin/bash

DATA=../dataset/sub_1000_openr1.parquet
OUTPUT_DIR=./tracking_results/
TEMPLATE=own

# Define arrays for models and their paths
MODEL_PATHS=(
  /jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
  # Add more model paths here
)

MODEL_NAMES=(
  Qwen-Math-7B
  # Add corresponding model names here
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
  
  CUDA_VISIBLE_DEVICES=1,2,3,4 python generate_acc.py \
    --n 8 \
    --model_path "$MODEL_PATH" \
    --input_file "$DATA" \
    --remove_system True \
    --output_file "$OUTPUT_DIR/sub_1000_openr1_${MODEL_NAME}_acc.parquet" \
    --template "$TEMPLATE"
  
  echo "Completed processing for $MODEL_NAME"
  echo ""
done

echo "All models processed successfully"