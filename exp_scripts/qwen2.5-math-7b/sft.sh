export MODEL_PATH=/jizhicfs/hymiezhao/models/Qwen2.5-Math-7B-16k-think
export DATA_DIR=./dataset/

export EXP_NAME=7b_SFT
export WANDB_PROJECT="rl-sft"

# data.val_files=$DATA_DIR/valid.parquet \

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.my_fsdp_sft_trainer \
    data.train_files=$DATA_DIR/sft.parquet \
    data.val_files=$DATA_DIR/sft_val.parquet \
    optim.lr=1e-5 \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=96 \
    data.micro_batch_size=48 \
    data.max_length=16384 \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    +trainer.do_validation=False \
    +trainer.save_checkpoint_steps=238 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir=./train_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.total_epochs=5 \
    trainer.logger=['console','wandb']

python /jizhicfs/hymiezhao/ml/busy.py