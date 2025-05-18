DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=/jizhicfs/hymiezhao/models/LUFFY-Qwen-Math-1.5B-Zero
MODEL_NAME=luffy-1.5B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-1.5B-16k-think
MODEL_NAME=Qwen-Math-1.5B
TEMPLATE=qwen
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SFT/global_step_477
MODEL_NAME=SFT-7B
TEMPLATE=own

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log