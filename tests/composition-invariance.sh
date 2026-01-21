#!/bin/bash
###########################################################
# Checking if algorithm composition works properly
#
#   algo2 * algo1(guarantee) == algo2(algo1(guarantee))
#
###########################################################

###########################################################
# Bellow choose one the supported compositions:
## Parallel + Parallel Composition
#   * td+bu-l-dfs
#   * td+bu-d-dfs
#   * td+bu-bfs
#
## Cyclic + Parallel Compositions
#   * c-bu-l+bu-l-dfs
#   * c-bu-l+bu-d-dfs
#   * c-bu-l+bu-bfs
#
#   * c-bu-d+bu-l-dfs
#   * c-bu-d+bu-d-dfs
#   * c-bu-d+bu-bfs
#
#   * c-td+bu-l-dfs
#   * c-td+bu-d-dfs
#   * c-td+bu-bfs
###########################################################
algo1="c-bu-d"
algo2="bu-d-dfs"
algo_comp=$algo1+$algo2

# Compute Algo. Comp.
python ../bin/parallelepipedonn.py \
-al $algo_comp \
-x ../data/seven/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-od $algo_comp \
-c 7 \
-si \
-ov

echo "##################################################################"

# Compute Algo. 1
python ../bin/parallelepipedonn.py \
-al $algo1 \
-x ../data/seven/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-od $algo1 \
-c 7 \
-si \
-ov

echo "##################################################################"

# Compute Algo 2
python ../bin/parallelepipedonn.py \
-al $algo2 \
-x ../data/seven/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-od $algo2 \
-lb ../data/seven/outputs/$algo1/7-1_lb.csv \
-ub ../data/seven/outputs/$algo1/7-1_ub.csv \
-c 7 \
-si \
-ov

echo "##################################################################"

# Both the above results should be equal
python ../bin/interval-eq.py \
-lb1 ../data/seven/outputs/$algo_comp/7-1_lb.csv \
-ub1 ../data/seven/outputs/$algo_comp/7-1_ub.csv \
-lb2 ../data/seven/outputs/$algo2/7-1_lb.csv \
-ub2 ../data/seven/outputs/$algo2/7-1_ub.csv \

echo $?