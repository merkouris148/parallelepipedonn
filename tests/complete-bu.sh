#!/bin/bash

python ../bin/parallelepipedonn.py \
-al complete-bu \
-v mara-complete \
-x ../data/sevens/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-c 7 \
-si \
-ov \
-mi 2000