#! /bin/bash

## compile
clear && \
mypyc \
./guarantees \
./geometry/ \
./verification/nn_verification.py \
./verification/marabou.py \
./algorithms/algorithms.py \
./algorithms/parallelepipedal.py \
./algorithms/cyclic.py

## change dir
# cd ./bin

# ## run
# #clear && \
# python ./parallelepipedonn.py \
# -x ../data/inputs/FashionMNIST/7_4.csv \
# -c 7 \
# -nn ../onnx_models/fashion_mnist_nn-64.onnx \
# -al complete-bu  \
# -d 0.1 \
# -v mara-complete \
# -mi 500 \
# -lg

# python ./parallelepipedonn.py \
# -x ../data/inputs/FashionMNIST/7_4.csv \
# -c 7 \
# -nn ../onnx_models/fashion_mnist_nn-64.onnx \
# -al td  \
# -d 0.1 \
# -v mara-sound \
# -mi 500 \
# -lg
#-r 0.2

