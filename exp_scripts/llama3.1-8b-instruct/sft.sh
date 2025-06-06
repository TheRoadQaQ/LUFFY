export MODEL_PATH=/jizhicfs/hymiezhao/models/Meta-Llama-3.1-8B-Instruct
export DATA_DIR=./dataset/

export EXP_NAME=llama_8_SFT
export WANDB_PROJECT="rl-sft"

# data.val_files=$DATA_DIR/valid.parquet \

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.my_fsdp_sft_trainer \
    data.train_files=$DATA_DIR/sft.parquet \
    data.val_files=$DATA_DIR/sft_val.parquet \
    optim.lr=1e-5 \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=128 \
    data.micro_batch_size=8 \
    data.max_length=16384 \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    +trainer.do_validation=False \
    +trainer.save_checkpoint_steps="[1071]" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir=./train_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.total_epochs=4 \
    trainer.logger=['console']

python /jizhicfs/hymiezhao/ml/busy.py