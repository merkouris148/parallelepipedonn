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

class TestFromArchitecture(unittest.TestCase):
    def __init__(self, methodName = "Testing MLP Generation from Architecture"):
        """
            We test the `FromArchitecture` constructor.
        """
        super().__init__(methodName)
        
        ## Parameters
        self.name           = "Test-FromArchitecture"    # name
        self.input_shape    = (5, 3)            # input shape
        self.architecture   = [15, 7, 5, 4, 2]  # architecture

        ## MLP
        self.nn = mlp.FromArchitecture(
            self.architecture,
            self.input_shape,
            self.name
        )


    def test00_report(self):
        print()
        self.nn.report()


    def test01_num_layers(self):
        self.assertTrue(len(self.nn.layers) == len(self.architecture))


    def test02_input_shape(self):
        self.assertTrue(
            self.nn.layers[0].input_shape[1:] ==\
            self.input_shape
        )


    def test03_weights_shape(self):
        ind                 = 0
        weight_shapes_ok    = True
        for layer in self.nn.layers:
            if len(layer.get_weights()) == 0: continue

            weight_shapes_ok = (
                layer.get_weights()[0].shape[0] ==\
                self.architecture[ind]
            )
            
            self.assertTrue(weight_shapes_ok)
            ind += 1
        
        last_layer  = self.nn.layers[-1]
        
        weight_shapes_ok = (
            last_layer.get_weights()[0].shape[1] ==\
            self.architecture[-1]
        )
        
        self.assertTrue(weight_shapes_ok)


    def test04_export2onnx(self):
        ## create onnx path
        onnx_path = "./" + self.name + ".onnx"
        if os.path.isfile(onnx_path): os.remove(onnx_path)

        ## export
        self.nn.export2onnx(onnx_path, False)

        ## check if the file has been created
        self.assertTrue(os.path.isfile(onnx_path))

        ## clean afterwards
        os.remove(onnx_path)
    

    def test05_export2csv(self):
        ## create csv dir
        csv_dir = "./" + self.name
        if os.path.isdir(csv_dir): shutil.rmtree(csv_dir)

        ## export
        self.nn.export2csv(csv_dir)

        ## check if the directory is created
        self.assertTrue(os.path.isdir(csv_dir))

        ## check if the parameter csvs are created
        ## Weights csvs
        W_csvs = [
            csv_dir + "/" + item
            for item in os.listdir(csv_dir)
            if
                os.path.isfile(csv_dir + "/" + item)
            and mlp.is_parameter_input(item)
            and item[0] == "W"
        ]
        W_csvs.sort()
        self.assertTrue(len(W_csvs) == len(self.architecture) - 1)

        ## Bias csvs
        b_csvs = [
            csv_dir + "/" + item
            for item in os.listdir(csv_dir)
            if
                os.path.isfile(csv_dir + "/" + item)
            and mlp.is_parameter_input(item)
            and item[0] == "b"
        ]
        b_csvs.sort()
        self.assertTrue(len(b_csvs) == len(self.architecture) - 1)

        ## Load the Weights and biases
        matfromcsv = lambda file: np.genfromtxt(file)
        Weights = list(map(matfromcsv, W_csvs))
        biases  = list(map(matfromcsv, b_csvs))

        ## check all weights to be correct
        ind = 0
        for layer in self.nn.layers:
            if layer.name == "flatten": continue

            self.assertTrue((layer.get_weights()[0] == Weights[ind]).all())
            self.assertTrue((layer.get_weights()[1] == biases[ind]).all())
            ind += 1

        ## clean afterwards
        shutil.rmtree(csv_dir)
    

    def test06_naive_semantic_eq(self):
        self.assertTrue(self.nn.semantic_eq(self.nn))



if __name__ == "__main__":
    unittest.main(verbosity=2)
    #pass



