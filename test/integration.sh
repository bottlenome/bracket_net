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

python -m scripts.train_cube params.batch_size=1024 params.num_epochs=${EPOCHS} \
params.num_layers=${NUM_LAYERS} params.d_model=16 params.n_head=1 \
data.size_max=1024

python -m scripts.train_cube params.batch_size=1024 params.num_epochs=${EPOCHS} \
params.num_layers=${NUM_LAYERS} params.d_model=16 params.n_head=1 \
model.name=up-causal-naive \
data.size_max=1024

python -m scripts.train_cube_ristricted data.name=StateDistanceLoader \
data.size_max=102400 params.num_epochs=${EPOCHS}

python -m scripts.train_cube_ristricted data.name=StateNextActionLoader \
data.size_max=102400 params.num_epochs=${EPOCHS}

echo "PASS"
