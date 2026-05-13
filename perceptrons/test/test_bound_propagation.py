import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
sys.path.append("..")

import numpy as np

import multilayer as mlp

class BoundPropagationTest(unittest.TestCase):
    def __init__(self, methodName = "Testing MLP Bound Propagation"):
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
        
        ## Verbosity
        self.verbose        = True

        ## Inputs
        self.input_shape    = (2, 2)
        domain1             = mlp.intervals.Interval(
                                np.zeros(self.input_shape),
                                np.ones(self.input_shape)
                            )
        domain2             = mlp.intervals.Interval(
                                -1.0 * np.ones(self.input_shape),
                                np.ones(self.input_shape)
                            )
        domain3             = mlp.intervals.Interval(
                                20.0 * np.ones(self.input_shape),
                                50.0 * np.ones(self.input_shape)
                            )
        domain4             = mlp.intervals.Interval(
                                -20.0 * np.ones(self.input_shape),
                                -50.0 * np.ones(self.input_shape)
                            )
        self.domains        = [
                                domain1,
                                domain2,
                                domain3,
                                domain4
                            ]

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


    def test00_report(self):
        print()
        self.nn.report()

    
    def test01_bound_propagation01(self):
        for domain in self.domains:
            I_hat, I = self.nn.propagate_bounds(domain)


            self.assertTrue(len(I_hat) == len(I) + 1)
            self.assertTrue(len(I_hat) == len(self.nn.layers))

            if not self.verbose: continue

            print("\nPost-Activation Bound:")
            for ind in range(len(I_hat)): print(I_hat[ind])

            print("\nPre-Activation Bound:")
            for ind in range(len(I)): print(I[ind])


    def test02_bound_propagation02(self):
        for domain in self.domains:
            I_hat, _ = self.nn.propagate_bounds(domain)

            for i in range(1, len(I_hat)): self.assertTrue(I_hat[i].is_positive())



if __name__ == "__main__":
    unittest.main(verbosity=2)