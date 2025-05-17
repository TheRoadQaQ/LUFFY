export MODEL_PATH=/data/malu/Qwen2.5-0.5B-Instruct
export DATA_DIR=./dataset/

export EXP_NAME=debug
export WANDB_PROJECT="rl-sft"

# data.val_files=$DATA_DIR/valid.parquet \

CUDA_VISIBLE_DIVICES=3,4 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.my_fsdp_sft_trainer \
    data.train_files=$DATA_DIR/sft.parquet \
    data.val_files=$DATA_DIR/sft_val.parquet \
    optim.lr=1e-6 \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=256 \
    data.max_length=8192 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs=10 \
    trainer.logger=['console']