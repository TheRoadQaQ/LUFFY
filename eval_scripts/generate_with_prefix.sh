DATA=../dataset/sub_1000_openr1.parquet
OUTPUT_DIR=./prefix_results/

MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
MODEL_NAME=Qwen-Math-7B
TEMPLATE=own
#TEMPLATE=qwen

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SFT/global_step_477
MODEL_NAME=SFT-7B
TEMPLATE=own

MODEL_PATH=/jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_SEMI_LUFFY/best/actor
MODEL_NAME=SEMI_LUFFY
TEMPLATE=own

for prefix in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "Running with prefix=$prefix"
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_vllm_with_prefix.py \
      --model_path $MODEL_PATH \
      --input_file $DATA \
      --remove_system True \
      --output_file $OUTPUT_DIR/prefix_${prefix}.jsonl \
      --template $TEMPLATE \
      --temperature 1.0 \
      --prefix_rate $prefix > $OUTPUT_DIR/prefix_${prefix}.log
done

python  /jizhicfs/hymiezhao/ml/busy.py