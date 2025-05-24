#!/bin/bash

OOD_DATA=../dataset/sub_1000_openr1.parquet
#TRAIN_DATA=../dataset/sub_8000_openr1.parquet

OUTPUT_DIR=./track_results/qwen-math-7b-sft
TEMPLATE=own

# 定义模型及其路径
MODEL_PATHS=(
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_sub_SFT/global_step_30
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_sub_SFT/global_step_60
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_sub_SFT/global_step_90
  /jizhicfs/hymiezhao/ml/reasoning/LUFFY/train_results/rl-sft/7b_sub_SFT/global_step_120
)

MODEL_NAMES=(
  Step_30
  Step_60
  Step_90
  Step_120
)

# 检查数组长度是否一致
if [ ${#MODEL_PATHS[@]} -ne ${#MODEL_NAMES[@]} ]; then
  echo "错误：MODEL_PATHS和MODEL_NAMES数组长度必须相同"
  exit 1
fi

# 定义GPU配对 (8个GPU分成4对)
GPU_PAIRS=(
  "0,1"   # 模型0使用GPU 0和1
  "2,3"   # 模型1使用GPU 2和3
  "4,5"   # 模型2使用GPU 4和5
  "6,7"   # 模型3使用GPU 6和7
)

# 检查GPU配对数量是否足够
if [ ${#GPU_PAIRS[@]} -lt ${#MODEL_PATHS[@]} ]; then
  echo "错误：GPU配对数量不足"
  exit 1
fi

# 运行模型的函数
run_model() {
  local index=$1
  local gpu_pair=$2
  
  MODEL_PATH=${MODEL_PATHS[$index]}
  MODEL_NAME=${MODEL_NAMES[$index]}
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动模型: $MODEL_NAME 使用GPU $gpu_pair"
  echo "模型路径: $MODEL_PATH"
  
  CUDA_VISIBLE_DEVICES=$gpu_pair python generate_acc.py \
    --n 8 \
    --temperature 1.0 \
    --model_path "$MODEL_PATH" \
    --input_file "$OOD_DATA" \
    --remove_system True \
    --output_file "$OUTPUT_DIR/sub_1000_openr1_${MODEL_NAME}_acc.parquet" \
    --template "$TEMPLATE"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已启动模型 $MODEL_NAME 在GPU $gpu_pair (PID: $!)"
  echo ""
}

# 并行启动所有模型
for i in "${!MODEL_PATHS[@]}"; do
  run_model $i ${GPU_PAIRS[$i]} &
done

# 等待所有后台进程完成
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有模型处理完成"

python /jizhicfs/hymiezhao/ml/busy.py