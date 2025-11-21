#!/usr/bin/env bash
source /opt/conda/etc/profile.d/conda.sh
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate finers
# cd /13105340479/finers
# pip install loguru

# python /15324359926/seg/HD_Reasoning/data_converter_fixed_1920_qa_with_best_region.py # 后面删除 仅需要执行一次


# pip install loguru
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="offline"

MODEL_PATH=ckpts/Qwen2.5-VL-7B-Instruct # replace it with your local file path
TRAIN_FILE=all_data/all_parquet/converted_data_1920gt_resize_box_qa_v1_before_revised_mask_final_version
# TRAIN_FILE=/15324359926/seg/HD_Reasoning/All_Data_HD_Reasoning/converted_data # 840 
SAVE_FREQ=40

GPU_NUM=2 #  4
BATCH_SIZE=8 # 16

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/seg_zero_7b_1920size_reward8_region_and_qa.yaml \
    data.val_files=None \
    data.train_files=${TRAIN_FILE} \
    data.rollout_batch_size=${BATCH_SIZE}\
    worker.actor.global_batch_size=${BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=5.0e-3 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.total_episodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_checkpoint_path=./workdir/${RUN_NAME} \
    trainer.resume_mode="auto" \
    trainer.max_steps=200