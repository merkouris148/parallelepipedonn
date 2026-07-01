#! /bin/bash

## compile
clear
## change dir
cd ./bin

python ./parallelepipedonn.py \
-x ../data/inputs/MNIST/7-4.csv \
-c 7 \
-nn ../onnx_models/mnist_nn-32.onnx \
-al td  \
-d 0.1 \
-v mara-sound \
-mi 4000 \
-lg