import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import sys
sys.path.append("../..")
import geometry.interval as intervals
import perceptrons.mnist as mnist

mnist_nn = mnist.MNIST()

mnist_nn.report()
# mnist_nn.export2csv("./", True)
# mnist_nn.export2onnx("./")
# mnist_nn.report2file()


# # from file
# mnist_nn_from_file = mnist.MNISTFromDir("./")
# mnist_nn_from_file.report()

# # check equality
# print(mnist_nn.semantic_eq(mnist_nn_from_file))

# # ## compute bounds
# bounds = mnist_nn_from_file.propagate_bounds(
#     intervals.Interval(np.zeros((28, 28)), np.ones((28, 28)))
# )

# for b in bounds:
#     print(b.lb)
#     print(b.ub)