#################################################
# Testing the MNISTNeuralNetwork class from
# the ../neural_network.py file
#################################################

#############
# Libraries #
#############

## Python libraries
# Importing parent directory class
# see: https://www.geeksforgeeks.org/python-import-from-parent-directory/
import sys
sys.path.append('..')
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore')

## 3rd party libraries
import numpy as np

## Custom libraries
import neural_network as nn
import montecarlo.utilities as mc

from pprint import pprint


############
# Constant #
############
# nn_name         = "CIFAR10"
nn_name         = "HandSign"
nn_name         = "MNIST"
# path to the weights
# weights_path_h5 = "../nn_weights/cifar10_nn-128-64-32.h5"
# onnx_path       = "../nn_weights/cifar10_nn-128-64-32.onnx"
# weights_path_h5 = "../nn_weights/hand_sign_nn-128-64-32.h5"
# onnx_path       = "../nn_weights/hand_sign_nn-128-64-32.onnx"
# weights_path_h5 = "../nn_weights/fashion_mnist_nn-64.h5"
# onnx_path       = "../nn_weights/fashion_mnist_nn-64.onnx"
weights_path_h5 = "../nn_weights/mnist_nn-32.h5"
onnx_path       = "../nn_weights/mnist_nn-32.onnx"
# number of test samples, to re-evaluate accuracy
testset_samples         = 1000
testset_samples_shuffle = False


if __name__ == "__main__":
    ###########################
    # Create a Neural Network #
    ###########################
    print("#### Testing", nn_name, "Neural Network ####\n")

    neural_net = nn.FashionMNISTNeuralNetwork(weights_path_h5)
    # neural_net = nn.HandSignNeuralNetwork(weights_path_h5)
    # neural_net = nn.MNISTNeuralNetwork(weights_path_h5)
    # neural_net = nn.CIFAR10NeuralNetwork(weights_path_h5)
    neural_net.initialization()
    neural_net.export2onnx(onnx_path)

    print("\nWeights loaded from: ", neural_net.where_weights)

    ## get dimentions
    row_dim     = neural_net.X_test[0][:].shape[0]
    column_dim  = neural_net.X_test[0][:].shape[1]
    num_classes = 26

    print("Input Dimension")
    print("\tRow Dimension: ", row_dim)
    print("\tColumn Dimension: ", column_dim)
    print("Output Dimension (# Classes)", num_classes)
    print("\tNumber of Classes: ", num_classes)

    ## Summary
    neural_net.model.summary()

    ## TF accuracy
    print("TF Test Set Accuracy: "      + str(neural_net.evaluate_test()))
    print("TF Training Set Accuracy: "  + str(neural_net.evaluate_train()))


    ## Re-evaluate Accuracy
    testset_sample_indices = list(range(testset_samples))
    if testset_samples_shuffle: np.random.shuffle(testset_sample_indices)

    experimental_acc, predictions = mc.evaluate(neural_net, neural_net.X_test, neural_net.Y_test)
    print("Experimental Accuracy: ", experimental_acc)


    ## Noise prediction analysis
    histogram, sample = mc.noise_analysis(
        np.zeros((row_dim, column_dim)),
        np.ones((row_dim, column_dim)),
        neural_net,
        num_classes
    )
    print("Noise Analysis: ", histogram)
