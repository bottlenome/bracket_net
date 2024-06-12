#!/usr/bin/bash -eu

EPOCHS=1
BATCH_SIZE=2
NUM_LAYERS=1

export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled

echo "Running integration tests"

python -m scripts.train \
model=bracket-naive params.batch_size=${BATCH_SIZE} gpt.num_layers=${NUM_LAYERS} \
gpt.n_head=1 model.mode=11_beam_search_optimized gpt.d_model=16 \
data.magnification=1 data.size_max=${BATCH_SIZE} params.num_epochs=${EPOCHS} \
params.enable_entropy_loss=False

python -m scripts.train \
model=bracket-naive params.batch_size=${BATCH_SIZE} gpt.num_layers=${NUM_LAYERS} \
gpt.n_head=1 model.mode=11_beam_search_optimized gpt.d_model=16 \
data.magnification=1 data.size_max=${BATCH_SIZE} params.num_epochs=${EPOCHS} \
params.enable_entropy_loss=True

python -m scripts.train \
model=reformer-naive params.batch_size=${BATCH_SIZE} gpt.num_layers=${NUM_LAYERS} \
gpt.n_head=1 gpt.d_model=16 \
data.magnification=1 data.size_max=${BATCH_SIZE} params.num_epochs=${EPOCHS}


echo "PASS"
