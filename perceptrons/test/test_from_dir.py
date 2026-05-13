import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
import shutil
sys.path.append("..")

import numpy as np

import multilayer as mlp

class FromDirTest(unittest.TestCase):
    def __init__(self, methodName = "Testing MLP Generation from Dir"):
        """
            We test the `FromDir` constructor.\\
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
        self.nn_from_weights = mlp.FromWeights(
            [self.W1, self.W2, self.W3],
            [self.b1, self.b2, self.b3],
            self.input_shape,
            "Test-FromWeights"
        )

        self.nn_from_dir = None


    def test00_report(self):
        print()
        self.nn_from_weights.report()
    

    def test01_export2csv(self):
        if os.path.isdir("./Test-FromDir"): shutil.rmtree("./Test-FromDir")

        self.nn_from_weights.export2csv("./Test-FromDir")
        self.assertTrue(os.path.isdir("./Test-FromDir"))
    

    def test02_import_csv(self):
        self.nn_from_dir = mlp.FromDir(
            "./Test-FromDir",
            self.input_shape
        )
        print()
        self.nn_from_dir.report()
        
        self.assertTrue(
            self.nn_from_weights.semantic_eq(
                self.nn_from_dir
            )
        )
    
    def test03_rmdir(self):
        ## clean afterwards
        shutil.rmtree("./Test-FromDir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
    #pass