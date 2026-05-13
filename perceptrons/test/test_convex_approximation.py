import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
sys.path.append("..")

import numpy as np

import multilayer as mlp
import convex

class ConvexApproxTest(unittest.TestCase):
    def __init__(self, methodName = "Testing Convex Approximation for Single Perceptron"):
        """
            We test the Convex Approximation for a Single Perceptron\\
            **Inputs:**
            **Parameters:**
            ```
                W1  =   [[1, 0, 0],    b1 = [1, 1]
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]]
            ```
        """
        super().__init__(methodName)
        
        ## Verbosity
        self.verbose        = True

        ## Inputs
        self.input_shape    = (2, 2)
        self.domain         = mlp.intervals.Interval(
                                -1.0 * np.ones(self.input_shape),
                                 1.0 * np.ones(self.input_shape)
                            )

        ## Parameters
        # W1
        self.W1 = np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]]
        )
        # b1
        self.b1 = np.array([0.1, -0.4, 0.7])

        ## MLP
        self.nn = mlp.FromWeights(
            [self.W1],
            [self.b1],
            self.input_shape,
            "Test-Single-Perceptron"
        )
        #self.nn.report()

        ## Convex MLP
        self.convex_nn = convex.ConvexApprox(
            self.nn,
            self.domain
        )
        #self.convex_nn.report()
    
    def test00_reports(self):
        self.nn.report()
        self.convex_nn.report()

    def test01_diff(self):
        x = np.zeros(self.nn.in_shape)

        s_relu = self.nn.scores(x)
        s_conv = self.convex_nn.scores(x)

        print("\n\nReLU value = ")
        print(s_relu)

        print("\n\nConvex value = ")
        print(s_conv)

        print("\n\ndifference = ")
        print(s_relu - s_conv)


    def test02_avg_divergance(self):
        N   = 1000
        Xs  = self.domain.random_points(N)

        Ss_relu = self.nn.scores(Xs)
        Ss_conv = self.convex_nn.scores(Xs)

        Diff = np.max(np.abs(Ss_relu - Ss_conv), 0)

        avg = np.sum(Diff) / len(Diff)
        print(avg)


if __name__ == "__main__":
    unittest.main(verbosity=2)