#############
# Libraries #
#############

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mnist
import convex as conv

import numpy as np

import sys
sys.path.append("..")
import geometry.interval as intervals
import montecarlo.utilities as utils

#############
# Constants #
#############

do_tests                = False
train_from_scratch      = False
reports                 = True
eval_conv_approx_err    = True
image_inputs_toggle     = True

########
# Code #
########

mnist_nn = None
if train_from_scratch:
    mnist_nn = mnist.MNIST()

    mnist_nn.report()
    mnist_nn.export2csv()
    mnist_nn.export2onnx()
    mnist_nn.report2file()
else:
    ## from file
    mnist_nn = mnist.MNISTFromDir("./MNIST-mlp")
    if reports: mnist_nn.report()

## input domain
positive_quadrant   = intervals.Interval(np.zeros((10, 1)), np.inf * np.ones((10, 1)))
domain              = intervals.Interval(np.zeros((28, 28)), np.ones((28, 28)))

conv_approx = conv.ConvexApprox(mnist_nn, domain)
if reports: conv_approx.report()




#######################################
# Evaluate Convex Approximation Error #
#######################################
if eval_conv_approx_err:
    num_of_rand_inputs  = 100_000
    seed                = 0
    err                 = 0

    print("Num. of Random Inputs:", num_of_rand_inputs)
    print("Seed:", seed)

    
    RandInputs  = None
    if image_inputs_toggle:
        RandInputs  = mnist_nn.random_points(num_of_rand_inputs, seed)
    else:
        RandInputs  = domain.random_points(num_of_rand_inputs, seed)
    Scores      = mnist_nn.scores(RandInputs)
    ConvScores  = conv_approx.scores(RandInputs)
    Diff        = np.abs(Scores - ConvScores)
    ErrVec      = np.max(Diff, 1)
    Err         = round(np.sum(ErrVec) / num_of_rand_inputs, 2)

    print("Average Error on Random Inputs:", Err)

    ConvBounds          = conv_approx.propagate_bounds(domain)
    ConvScoresDomain    = ConvBounds[-1]
    conv_diameter       = round(ConvScoresDomain.diameter(), 2)
    Bounds              = mnist_nn.propagate_bounds(domain)
    ScoresDomain        = Bounds[-1]
    diameter            = round(ScoresDomain.diameter(), 2)
    # I should delete the last dense non-relu layer
    # for the scores to be in the positive_quadrant
    #ScoresDomain.intersect(positive_quadrant)

    # print("Domain Interval")
    # print("Lower Bound")
    # print(ScoresDomain.lb)
    # print("Upper Bound")
    # print(ScoresDomain.ub)
    # print("Scores Domain Diameter:", diameter)

    print("Conv. Domain Interval")
    print("Lower Bound")
    print(ConvScoresDomain.lb)
    print("Upper Bound")
    print(ConvScoresDomain.ub)
    print("Scores Domain Diameter:", diameter)

    ratio = round(Err / conv_diameter, 2)
    print("Error Ratio:", ratio)




################
# Some Testing #
################

    L = len(mnist_nn.layers_list)
    num_errors = 0
    for k in range(0, L, 2):
        if (conv_approx.weights[2*k].numpy() != mnist_nn.weights[k//2].numpy()).any():
            num_errors +=1
        if (conv_approx.weights[2*k + 1].numpy() != mnist_nn.weights[k//2 + 1].numpy()).any():
            num_errors +=1

    print("Mismatches in Weight Matrices:", num_errors)


if do_tests:
    ##########################
    # Count Mismatches no. 1 #
    ##########################
    N = 10_000
    RandomImages = domain.random_points(N, 0)
    #print(RandomImages.shape)
    Y_conv      = conv_approx.predict(RandomImages)
    Y_relu      = mnist_nn.predict(RandomImages)
    #print(Y_conv[0:1000])
    #print(Y_relu[0:1000])
    mismatches  = np.sum(Y_conv != Y_relu)
    print(mismatches / N)


    ##########################
    # Count Mismatches no. 2 #
    ##########################
    #print(RandomImages.shape)
    Y_conv_2      = conv_approx.predict(mnist_nn.X_test[0:N, :, :])
    Y_relu_2      = mnist_nn.predict(mnist_nn.X_test[0:N, :, :])
    #print(Y_conv[0:1000])
    #print(Y_relu[0:1000])
    mismatches_2  = np.sum(Y_conv_2 != Y_relu_2)
    print(mismatches_2 / N)


    ##########################
    # Count Mismatches no. 3 #
    ##########################
    #print(RandomImages.shape)
    Y_conv_3      = conv_approx.predict(mnist_nn.X_train[0:N, :, :])
    Y_relu_3      = mnist_nn.predict(mnist_nn.X_train[0:N, :, :])
    #print(Y_conv[0:1000])
    #print(Y_relu[0:1000])
    mismatches_3  = np.sum(Y_conv_3 != Y_relu_3)
    print(mismatches_3 / N)


    #####################
    # Test Bounds no. 1 #
    #####################
    Bounds  = mnist_nn.propagate_bounds(domain)
    Scores  = mnist_nn.scores(RandomImages)
    successful_bounds = 0
    for i in range(Scores.shape[0]):
        if Scores[i,:].reshape(Scores[i,:].shape[0], 1) in Bounds[-1]:
            successful_bounds += 1

    print(successful_bounds / N)



    ####################
    # Test Bounds no. 2#
    ####################
    Bounds_conv  = conv_approx.propagate_bounds(domain)
    Scores_conv  = conv_approx.scores(RandomImages)
    successful_bounds_conv = 0
    for i in range(Scores_conv.shape[0]):
        if Scores_conv[i,:].reshape(Scores_conv[i,:].shape[0], 1) in Bounds_conv[-1]:
            successful_bounds_conv += 1

    print(successful_bounds_conv / N)

    ####################
    # Test Bounds no. 3#
    ####################
    print(np.sum(Bounds[-1] != Bounds_conv[-1]) / Bounds[-1].row_dim)