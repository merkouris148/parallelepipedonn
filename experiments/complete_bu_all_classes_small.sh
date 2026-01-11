#!/bin/bash

###########################################################
# A BASH script for perfoming *sequentially* the
# experiments for all the datasets that include a single
# class.
###########################################################

## Fixed variables
onnx_path="../nn_weights/mnist_description.onnx"
num_threads=20
max_it=2000

## Inputs
input="../data/all_classes_small/inputs"

## Outputs
output="../data/all_classes_small/outputs"
method="complete-bu"
output_subdir=$output"/"$method"/"
## Experiments
nohup python experiments_script.py $input $onnx_path $num_threads $max_it $method > $output_subdir"/"$method"_nohup.out"
