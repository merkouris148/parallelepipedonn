#!/bin/bash

###########################################################
# A BASH script for perfoming *sequentially* the
# experiments for all the datasets that include a single
# class.
###########################################################

## Fixed variables
onnx_path="../nn_weights/hand_sign_nn-128-64-32.onnx"
num_threads=35

## Inputs
input="../data/inputs/HandSign"

## Outputs
output="../data/outputs/HandSign"

## Timeout
timeout=60

# 10 Max. It.
max_it=10   # iterations for each coordinate
methods=(
    "c-bu-d"
    "bu-d-dfs"
    "complete-c-d-bu"
)

## Experiments
for method in "${methods[@]}"; do
    echo "Applying "$method" to "$input
    output_subdir=$output"/"$method"/"
    if [ ! -d $output_subdir ]; then
        mkdir $output_subdir
    fi
    python experiments_script.py $input $onnx_path $num_threads $max_it $timeout $method > $output_subdir"/"$method"_nohup.out"
    echo "Experiments with "$method" in "$input" done! Exit code: "$?
done


## 10_000 Max. It.
max_it=10000   # iterations for each coordinate
methods=(
    "td"
    "complete-bu"
)

## Experiments
for method in "${methods[@]}"; do
    echo "Applying "$method" to "$input
    output_subdir=$output"/"$method"/"
    if [ ! -d $output_subdir ]; then
        mkdir $output_subdir
    fi
    python experiments_script.py $input $onnx_path $num_threads $max_it $timeout $method > $output_subdir"/"$method"_nohup.out"
    echo "Experiments with "$method" in "$input" done! Exit code: "$?
done



## Composition/10 Max. It.
# max_it=10   # iterations for each coordinate
# method="bu-d-dfs"
# bounds="../data/outputs/FashionMNIST/td"

# ## Experiments
# echo "Applying "$method" to "$input
# output_subdir=$output"/"$method"_comp/"
# if [ ! -d $output_subdir ]; then
#     mkdir $output_subdir
# fi

# python experiments_script.py\
#  $input\
#  $onnx_path\
#  $num_threads\
#  $max_it\
#  $timeout\
#  $method\
#  $bounds\
#  > $output_subdir"/"$method"_nohup.out"

# echo "Experiments with "\
#  $method" in "$input\
#  " done! Exit code: "$?