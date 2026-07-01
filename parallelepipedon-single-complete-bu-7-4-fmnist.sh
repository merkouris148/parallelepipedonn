#! /bin/bash

## compile
clear
## change dir
cd ./bin

python ./parallelepipedonn.py \
-x ../data/inputs/FashionMNIST/7_4.csv \
-c 7 \
-nn ../onnx_models/fashion_mnist_nn-64.onnx \
-al complete-bu \
-d 0.1 \
-v mara-complete \
-mi 100 \
-lg