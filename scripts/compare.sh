#!/bin/bash -eu
# experiment_name="diagonal_0"
experiment_names=("goal_estimation_from_state"
                  "entropy_loss"
                  )
# branch_name="diagonal_1"
# branch_name="main"
num_epochs=50

for branch_name in "${experiment_names[@]}"
do
experiment_name=$branch_name

git checkout $branch_name
for i in {1..2}
do
    echo "Running experiment ${experiment_name} $i"
    cmd="python -m scripts.train model=bracket-naive experiment_name=${experiment_name}_${i} params.batch_size=10 gpt.num_layers=3 gpt.n_head=1 model.mode=11_beam_search_optimized gpt.d_model=128 data.magnification=1 params.num_epochs=${num_epochs} -m"
    bash -c "$cmd"
    # break if sigint received
    if [ $? -eq 130 ]; then
        break
    fi
done

done