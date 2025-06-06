DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results/

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SFT/epoch3/
MODEL_NAME=SFT-7B-epoch3
TEMPLATE=own

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE

python /jizhicfs/hymiezhao/ml/busy.py