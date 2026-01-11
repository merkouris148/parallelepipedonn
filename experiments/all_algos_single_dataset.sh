#!/bin/bash

###########################################################
# A BASH script for perfoming *sequentially* the
# experiments for all the datasets that include a single
# class.
###########################################################

## Fixed variables
onnx_path="../nn_weights/mnist_description.onnx"
num_threads=19
max_it=10

## Inputs
input="../data/all_classes_small/inputs"

## Outputs
output="../data/all_classes_small/outputs"

methods=(
    "c-bu-l+bu-l-dfs"
    "c-bu-l+bu-d-dfs"
    "c-bu-l+bu-bfs"
    "c-bu-d+bu-l-dfs"
    "c-bu-d+bu-d-dfs"
    "c-bu-d+bu-bfs"
    "c-td+bu-l-dfs"
    "c-td+bu-d-dfs"
    "c-td+bu-bfs"
)

## Experiments
for method in "${methods[@]}"; do
    echo "Applying "$method" to "$input
    output_subdir=$output"/"$method"/"
    if [ ! -d $output_subdir ]; then
        mkdir $output_subdir
    fi
    python experiments_script.py $input $onnx_path $num_threads $max_it $method > $output_subdir"/"$method"_nohup.out"
    echo "Experiments with "$method" in "$input" done! Exit code: "$?
done
