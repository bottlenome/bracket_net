#!/bin/bash -eu
experiment_name="diagonal_0"
branch_name="main"
num_epochs=50

git checkout $branch_name
for i in {1..8}
do
    echo "Running experiment $i"
    cmd="python -m scripts.train model=bracket-naive experiment_name=${experiment_name}_${i} params.batch_size=10 gpt.num_layers=3 gpt.n_head=1 model.mode=11_beam_search_optimized gpt.d_model=128 data.magnification=1 params.num_epochs=${num_epochs} -m"
    bash -c "$cmd"
    sleep 10
    # break if sigint received
    if [ $? -eq 130 ]; then
        break
    fi
done