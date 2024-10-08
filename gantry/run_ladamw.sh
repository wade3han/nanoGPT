#!/bin/bash
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6
NAME=ladamw

export WANDB_ENTITY=seungjuhan3
gantry run --beaker-image seungjuh/open-instruct-public-240806-preview --venv base --name $NAME \
  --cluster ai2/pluto-cirrascale --workspace ai2/safety --pip requirements.txt \
  --gpus $NUM_GPUS --priority normal \
  --preemptible --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret HF_TOKEN=HUGGING_FACE_HUB_TOKEN \
  --env WANDB_ENTITY=seungjuhan3 --env-secret OPENAI_API_KEY=openai_api_key --budget ai2/oe-adapt -- \
  torchrun --standalone --nproc_per_node=4 train.py /net/nfs.cirrascale/mosaic/seungjuh/nanoGPT/config/train_gpt2_4gpus_ladamw.py
