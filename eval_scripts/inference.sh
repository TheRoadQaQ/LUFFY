DATA=../dataset/valid.all.parquet
OUTPUT_DIR=./results

MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
MODEL_NAME=Qwen-Math-7B-qwen-template
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_LUFFY_TEST/best/actor/
MODEL_NAME=LUFFY-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE/best/actor/
MODEL_NAME=INTERLEAVE-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_random_buffer/best/actor/
MODEL_NAME=INTERLEAVE-random-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SFT/epoch2.5/
MODEL_NAME=SFT-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_on_policy/best/actor/
MODEL_NAME=RL-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_all_buffer/best/actor/
MODEL_NAME=INTERLEAVE-all-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_INTERLEAVE_random2_buffer/best/actor/
MODEL_NAME=INTERLEAVE-uniform-7B
TEMPLATE=own

CUDA_VISIBLE_DEVICES=4,5,6,7 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python /jizhicfs/hymiezhao/ml/busy.py