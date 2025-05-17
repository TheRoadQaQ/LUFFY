DATA=../dataset/sub_1000_openr1.parquet
OUTPUT_DIR=./results/

MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
MODEL_NAME=Qwen-Math-7B
TEMPLATE=own

CUDA_VISIBLE_DEVICES=1,2,3,4 python generate_acc.py \
  --n 8 \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/sub_1000_openr1_${MODEL_NAME}_acc.parquet \
  --template $TEMPLATE