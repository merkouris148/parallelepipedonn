#! /bin/bash

# Clearing Screen
clear

# Inputs
input="../data/inputs/MNIST"
# ONNX Model
onnx_path="../onnx_models/mnist_nn-32.onnx"
# Number of Threads
num_threads=35
# Maximum Iterations
max_it=1000
# Timeout
timeout=60
# Algorithm
method="complete-bu"

# Outputs
mydateformat=$(date "+%d-%m-%Y")
output_dir="../data/outputs/MNIST/"$method"-"$mydateformat
output_file=$output_dir"/"$method"-"$mydateformat"_nohup.out"

# Create path
mkdir -p $output_dir
touch $output_file

# Run Experiments
python experiments_script.py \
$input \
$onnx_path \
$num_threads \
$max_it \
$timeout \
$method > \
$output_file