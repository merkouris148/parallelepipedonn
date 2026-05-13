import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
sys.path.append("..")

import numpy as np

import multilayer as mlp

class FromWeightsTest(unittest.TestCase):
    def __init__(self, methodName = "Testing MLP Generation from Weights"):
        """
            We test the `FromWeights` constructor.\\
            **Inputs:**
            **Parameters:**
            ```
                W1  =   [[1, 0, 0],    b1 = [1, 1]
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]]
            
                W2  =   [[0.1, 0, 0],  b2 = [-1, 1]
                        [0,    2, 0],
                        [0,    0, 2]]
                
                W3  =   [[1, 0],       b3 = [1, -3]
                        [0,  0],
                        [0,  1]]
            ```
        """
        super().__init__(methodName)
        
        ## Inputs
        self.input_shape    = (2, 2)
        self.seed           = 0
        self.N              = 100

        self.rnjeasus    = np.random.default_rng(self.seed)
        self.Xs          = []
        for i in range(self.N):
            X = self.rnjeasus.uniform(
                -1.0 * np.ones(self.input_shape),
                np.ones(self.input_shape)
            )
            self.Xs.append(X)

        ## Parameters
        # W1
        self.W1 = np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]]
        )
        # b1
        self.b1 = np.array([1, 1, 1])

        # W2
        self.W2 = np.array(
            [[0.1, 0, 0],
            [0,    2, 0],
            [0,    0, 2]]
        )
        # b2
        self.b2 = np.array([-1, 1, -2])

        # W3
        self.W3 = np.array(
            [[1, 0],
            [0,  0],
            [0,  1]]
        )
        # b3
        self.b3 = np.array([1, -3])

        ## MLP
        self.nn = mlp.FromWeights(
            [self.W1, self.W2, self.W3],
            [self.b1, self.b2, self.b3],
            self.input_shape,
            "Test-FromWeights"
        )

    def _linalg_mlp(self, x):
        x_flat = x.flatten()

        ## True Value
        z1      = np.matmul(x_flat, self.W1) + self.b1
        z1_hat  = np.maximum(z1, np.zeros(3))

        z2      = np.matmul(z1_hat, self.W2) + self.b2
        z2_hat  = np.maximum(z2, np.zeros(3))
        
        z3      = np.matmul(z2_hat, self.W3) + self.b3
        z3_hat  = np.maximum(z3, np.zeros(2))

        return z3_hat


    def test00_report(self):
        print()
        self.nn.report()
    
    def test01_evaluation(self):
        for i in range(self.N):
            ## LinAlg value
            z_linalg = self._linalg_mlp(self.Xs[i])

            ## TF value
            z_tf = self.nn.scores(self.Xs[i])

            ##check the truth!
            self.assertTrue((z_linalg == z_tf).all())



if __name__ == "__main__":
    unittest.main(verbosity=2)