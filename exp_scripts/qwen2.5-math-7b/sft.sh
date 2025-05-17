export MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
export DATA_DIR=./dataset/

export EXP_NAME=7b_SFT
export WANDB_PROJECT="rl-sft"

# data.val_files=$DATA_DIR/valid.parquet \

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/openr1.parquet \
    optim.lr=1e-6 \
    data.prompt_key=prompt \
    data.response_key=target \
    data.train_batch_size=256 \
    data.max_length=8192 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs=10 \
    trainer.logger=['console','wandb']