#!/bin/bash
###########################################################
# Checking if an algorithm that returns a maximal
# guarantee adhears to lazynes, namely:
#
#   max_algo(max_algo(guarantee)) = max_algo(guarantee)
#
###########################################################

###########################################################
# Bellow choose one of the maximal algorithms:
## Parallelepipedal Args 
#   * bu-l-dfs
#   * bu-d-dfs
#   * bu-bfs
#
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
algo="bu-d-dfs"

# Compute Bottom-Up Dich. DFS on seven
python ../bin/parallelepipedonn.py \
-al $algo \
-x ../data/seven/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-od $algo \
-c 7 \
-si\

# Compute Bottom-Up Dich. DFS on the results of seven
python ../bin/parallelepipedonn.py \
-al $algo \
-x ../data/seven/inputs/7-1.csv \
-nn ../nn_weights/mnist_description.onnx \
-od $algo-redirect \
-lb ../data/seven/outputs/$algo/7-1_lb.csv \
-ub ../data/seven/outputs/$algo/7-1_ub.csv \
-c 7 \
-si\

# Both the above results should be equal
python ../bin/interval-eq.py \
-lb1 ../data/seven/outputs/$algo/7-1_lb.csv \
-ub1 ../data/seven/outputs/$algo/7-1_ub.csv \
-lb2 ../data/seven/outputs/$algo-redirect/7-1_lb.csv \
-ub2 ../data/seven/outputs/$algo-redirect/7-1_ub.csv \