import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
sys.path.append("..")
import shutil

import numpy as np

import multilayer as mlp

class TestFromRandom(unittest.TestCase):
    def __init__(self, methodName = "Testing MLP Generation from Radom Weights"):
        """
            We test the `FromRandom` constructor.
        """
        super().__init__(methodName)
        
        ## Parameters
        self.name           = "Test-FromWeights"    # name
        self.seed           = 0                     # RNG seed
        self.input_shape    = (5, 3)                # input shape
        self.architecture   = [15, 7, 5, 4, 2]      # architecture

        ## MLP
        self.nn = mlp.FromRandom(
            self.architecture,
            self.input_shape,
            self.seed,
            self.name
        )


    def test00_report(self):
        print()
        self.nn.report()
    

if __name__ == "__main__":
    unittest.main(verbosity=2)