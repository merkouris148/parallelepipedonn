#!/bin/bash

###########################################################
# A BASH script for perfoming *sequentially* the
# experiments for all the datasets that include a single
# class.
###########################################################

## Fixed variables
onnx_path="../nn_weights/mnist_description.onnx"
num_threads=20
max_it=10
method="c-d-bu"

## Inputs
input0="../data/zeros/inputs"
input1="../data/ones/inputs"
input2="../data/twos/inputs"
input3="../data/threes/inputs"
input4="../data/fours/inputs"
input5="../data/fives/inputs"
input6="../data/sixs/inputs"
input7="../data/sevens/inputs"
input8="../data/eights/inputs"
input9="../data/nines/inputs"

## Outputs
output0="../data/zeros/outputs"
outut1="../data/ones/outputs"
output2="../data/twos/outputs"
output3="../data/threes/outputs"
output4="../data/fours/outputs"
output5="../data/fives/outputs"
output6="../data/sixs/outputs"
output7="../data/sevens/outputs"
output8="../data/eights/outputs"
output9="../data/nines/outputs"

## Experiments
echo "Experiments from "$input0
nohup python experiments_script.py $input0 $onnx_path $num_threads $max_it $method > $output0"/"$method"_nohup.out"
echo "Experiments from "$input1" done! Exit code: "$?

echo "Experiments from "$input1
nohup python experiments_script.py $input1 $onnx_path $num_threads $max_it $method > $output1"/"$method"_nohup.out"
echo "Experiments from "$input1" done! Exit code: "$?

echo "Experiments from "$input2
nohup python experiments_script.py $input2 $onnx_path $num_threads $max_it $method > $output2"/"$method"_nohup.out"
echo "Experiments from "$input2" done! Exit code: "$?

echo "Experiments from "$input3
nohup python experiments_script.py $input3 $onnx_path $num_threads $max_it $method > $output3"/"$method"_nohup.out"
echo "Experiments from "$input3" done! Exit code: "$?

echo "Experiments from "$input4
nohup python experiments_script.py $input4 $onnx_path $num_threads $max_it $method > $output4"/"$method"_nohup.out"
echo "Experiments from "$input4" done! Exit code: "$?

echo "Experiments from "$input5
nohup python experiments_script.py $input5 $onnx_path $num_threads $max_it $method > $output5"/"$method"_nohup.out"
echo "Experiments from "$input5" done! Exit code: "$?

echo "Experiments from "$input6
nohup python experiments_script.py $input6 $onnx_path $num_threads $max_it $method > $output6"/"$method"_nohup.out"
echo "Experiments from "$input6" done! Exit code: "$?

echo "Experiments from "$input7
nohup python experiments_script.py $input7 $onnx_path $num_threads $max_it $method > $output7"/"$method"_nohup.out"
echo "Experiments from "$input7" done! Exit code: "$?

echo "Experiments from "$input8
nohup python experiments_script.py $input8 $onnx_path $num_threads $max_it $method > $output8"/"$method"_nohup.out"
echo "Experiments from "$input8" done! Exit code: "$?

echo "Experiments from "$input9
nohup python experiments_script.py $input9 $onnx_path $num_threads $max_it $method > $output9"/"$method"_nohup.out"
echo "Experiments from "$input9" done! Exit code: "$?