#############
# Libraries #
#############

## Python Libraries
#from pprint import pprint
import os.path
import re

## typing
import typing as t

## 3rd Party Libraries
# TensorFlow
import torch as pt

import numpy as np

## Custom Libraries
import sys
sys.path.append("..")
import geometry.interval as intervals
import perceptrons.abstract as abstractMLP


###########
# Classes #
###########
class PTMLPBaseClass(
        abstractMLP.MLPAbstract,
        pt.nn.Module
    ):
    """
    Implementing the ``MLPBaseClass`` in PyTorch.
    """
    def __init__(self, name: str = "multilayered-perceptron", activation = True):

        ## Init super class
        self.name = name

        ## Architecture
        self.architecture:  t.List[int]         = []    # layer dimentions (shapes)
        self.num_hid_layers:int                 = 0     # depth = len(architecture)
        self.in_shape:      t.Tuple[int]        = None  # the shape on the inputs before flattening.
        self.out_dim:       int                 = None  # architecture[-1]


        ## Datasets (these will be instantiated after the training)
        self.X_train:   np.ndarray    = None
        self.X_test:    np.ndarray    = None
        self.Y_train:   np.ndarray    = None
        self.Y_test:    np.ndarray    = None


        ## From file
        self.from_file:     bool      = False
        self.filepath:      str       = None

        ## activation
        self.activation = activation


        #######################################
        # Additional data members for PyTorch #
        #######################################

        self.loss:  t.Callable[[pt.Tensor, pt.Tensor], pt.Tensor]   = None
    
    ##################################
    # Additional Methods for PyTorch #
    ##################################

    def get_layers(self):
        results = []
        hooks   = []

        def make_hook(name):
            def hook_fn(module, input, output):
                results.append((
                    name,
                    tuple(input[0].shape),
                    tuple(output.shape)
                ))
            return hook_fn

        for name, layer in self.named_modules():
            if len(list(layer.children())) == 0:
                hooks.append(layer.register_forward_hook(make_hook(name)))

        with pt.no_grad():
            self(np.ones(self.in_shape))

        for hook in hooks:
            hook.remove()

        return results
  

    ##########################
    # Methods of AbstractMLP #
    ##########################

    ## Evaluation
    def adhoc_loss(
            self,
            X: np.ndarray = None,
            Y: np.ndarray = None, 
            verbose: int = 0
        ) -> float:
        ## Input Checks
        #assert X != None and Y != None or X == None and Y == None
        if X is not None and Y is not None:
            assert X.shape[0]   == Y.shape[0]
            assert X.shape[1:]  == self.in_shape


        ## Evaluation Data
        X = self.X_test if X is None else X
        Y = self.Y_test if Y is None else Y

        with pt.no_grad():
            current_loss = round(self.loss(self(X), Y), 4)
            return current_loss
    

    def accuracy(
            self,
            X: np.ndarray   = None,
            Y: np.ndarray   = None, 
            verbose: int    = 0
        ) -> float:
        ## Input Checks
        #assert X != None and Y != None or X == None and Y == None
        if X is not None and Y is not None:
            assert X.shape[0]   == Y.shape[0]
            assert X.shape[1:]  == self.in_shape
        

        ## Evaluation Data
        X = self.X_test if X is None else X
        Y = self.Y_test if Y is None else Y

        
        ## PyTorch Commands ##
        self.eval()

        with pt.no_grad():
            Scores          = self(X)                   # raw logits
            Preds           = pt.argmax(Scores, dim=1)  # predicted classes
            total_correct   = (Preds == Y).sum().item()
            total_samples   = Y.size(0)

            return total_correct / total_samples
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        ## Input Checks
        assert X is not None
        if len(X.shape) == len(self.in_shape): X = np.expand_dims(X, axis=0)

        ## PyTorch Commands ##
        self.eval()

        with pt.no_grad():
            Scores  = self(X)                   # raw logits
            Preds   = pt.argmax(Scores, dim=1)  # predicted classes

            return Preds

    def scores(self, X: np.ndarray) -> np.ndarray:
        ## Input Checks
        assert X is not None
        if len(X.shape) == len(self.in_shape): X = np.expand_dims(X, axis=0)
        
        ## PyTorch Commands ##
        self.eval()

        with pt.no_grad():
            Scores  = self(X)   # raw logits

            return Scores
        
    
    ## Report
    def __str__(self) -> str:
        init_method = "Training" if not self.from_file else "From File (" + self.filepath + ")"

        s =    "================================================================="  + "\n"
        s +=     "~~ Reporting ~~"                                                  + "\n"
        s +=    "=================================================================" + "\n"
        s +=    "Parameters:"                                                       + "\n"
        s +=    "_________________________________________________________________" + "\n"
        s +=    f"{'':<4}{'Name:':<19}{self.name}"                                  + "\n"
        s +=    f"{'':<4}{'In. Shape.:':<19}{str(self.in_shape)}"                   + "\n"
        s +=    f"{'':<4}{'Out Dim.:':<19}{str(self.out_dim)}"                      + "\n"
        s +=    f"{'':<4}{'Architercture:':<19}{str(self.architecture)}"            + "\n"
        s +=    f"{'':<4}{'Init.:':<19}{init_method}"                               + "\n"
        activation_method = "ReLU" if self.activation else "Linear"
        s +=    f"{'':<4}{'Activ. Func.:':<19}{activation_method}"                  + "\n"
        s +=    "=================================================================" + "\n"
        s +=    "Layers:"                                                           + "\n"
        s +=    "_________________________________________________________________" + "\n"
        s +=    f"{'':<4}{'Name':<12}{'In. Shape':<17}{'Out. Shape':<20}{'Weight Shape':<30}"        + "\n"
        for layer in self.get_layers():
            s += f"{'':<4}{layer[0]:<12}"
            s += f"{str(layer[1]):<17}"
            s += f"{str(layer[2]):<20}"
            # if len(layer.get_weights()) == 0:   s += f"{str(layer.get_weights()):<30}" + "\n"
            # else:                               s += f"{str(layer.get_weights()[0].shape):<30}" + "\n"
        s +=    "=================================================================" + "\n"
        s +=    "Metrics:"                                                                          + "\n"
        s +=    "_________________________________________________________________" + "\n"
        if self.X_train is not None or self.X_test is not None:
            s +=    f"{'':<4}{'Train Loss:':<19}{str(self.adhoc_loss(self.X_train, self.Y_train))}"     + "\n"
            s +=    f"{'':<4}{'Train Accuracy:':<19}{str(self.accuracy(self.X_train, self.Y_train))}"   + "\n"
            s +=    f"{'':<4}{'Test Loss:':<19}{str(self.adhoc_loss(self.X_test, self.Y_test))}"        + "\n"
            s +=    f"{'':<4}{'Test Accuracy:':<19}{str(self.accuracy(self.X_test, self.Y_test))}"      + "\n"
            s +=    "=================================================================" + "\n\n\n"
        else:
            s +=    f"{'':<4}{'Train Loss:':<19}{'(not initialized)'}"     + "\n"
            s +=    f"{'':<4}{'Train Accuracy:':<19}{'(not initialized)'}" + "\n"
            s +=    f"{'':<4}{'Test Loss:':<19}{'(not initialized)'}"      + "\n"
            s +=    f"{'':<4}{'Test Accuracy:':<19}{'(not initialized)'}"  + "\n"
            s +=    "=================================================================" + "\n\n\n"
        
        return s
    
    def report(self) -> None:
        print(self)

    ## I/O
    def report2file(
            self,
            # filepath:str = None,
            overwrite:bool  = True
        ) -> None:

        report_dir = "./" + self.name
        if not os.path.isdir(report_dir): os.makedirs(report_dir)

        report_path = report_dir + "/" + self.name + ".out"
        if not os.path.isfile(report_path) or overwrite:
            f_desc = open(report_path, "w")
            f_desc.write(str(self))
            f_desc.close()
    
    ## Export
    def export2onnx(
            self,
            onnx_path:str   = None,
            overwrite:bool  = False
        ) -> None:

        ## default onnx_path
        onnx_dir = "./" + self.name
        if not os.path.isdir(onnx_dir): os.makedirs(onnx_dir)
        onnx_path = onnx_dir + "/" + self.name + ".onnx" if onnx_path is None else onnx_path

        ## check the suffix
        assert onnx_path.split(".")[-1] == "onnx",\
        "Error: wrong suffix! Given suffix: `." + str(onnx_path.split(".")[-1])\
        + "`, correct suffix should be `.onnx`."

        ## export
        if not os.path.isfile(onnx_path) or overwrite:
            
            ## PyTorch Code ##
            self.eval()  # always set to eval before exporting

            pt.onnx.export(
                self,                       # model to export
                np.ones(self.in_shape),     # sample input (defines input shape)
                onnx_path,                  # output file path
                input_names=["input"],      # name for the input node
                output_names=["output"],    # name for the output node
                opset_version=17,           # ONNX opset version (17 is a safe
                                            # modern choice)
            )