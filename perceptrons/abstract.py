#############
# Libraries #
#############

import typing as t
import numpy as np


#############
# Constants #
#############

default_delim = " "


###########
# Classes #
###########

class MLPAbstract(t.Protocol):
    """
    The Base Class for Multilayered Perceptrons. This class models a specific
    class of NN of the form: ::
    
        [Flatten, input_shpae],
        [Dense, ReLU, dim_0],
        [Dense, ReLU, dim_1],
        ...
        [Dense, ReLU, dim_(L-1)]

    This defines a L-layered MLP of the form ``Z:IR^(dim_0) --> IR^(d_(L-1))``.
    
    **Data Members:**

    * ``architecture: List[int]``, a list containing the dimensions of each layer.
    * ``num_layers: int``, the number of layers.
    * ``input_shape: Tuple[int]``, the shape on the inputs *before* flattening.
    * ``out_dim: int``, the output dimension.
    * ``W: List[ndarray]``, the list of weights.
    * ``b: List[ndarray]``, the list of biases. Naturally.
    * ``X_train, X_test: ndarray``, the training datapoints.

    **Invariants:**

    * ``architecture = [dim_0, dim_1, ..., dim_(L-1)]``
    * ``num_layers == len(architecture)``
    * ``out_dim == architecture[-1]``
    * ``len(W) == len(b) == len(architecture)``
    * ``X_tain.shape == (n, input_shape)``, where ``n`` is the number of training samples.
    * ``X_test.shape == (m, input_shape)``, where ``n`` is the number of test samples.
    * ``Y_train.shape == (n,)`` and ``Y_test.shape == (m,)``.
    
    **Notes:**

    *All* the above data members should be properly initialized *before* the use\
    of the provided methods. Otherwise, the methods behavior is unkown.

    ^^^^^^^
    Methods
    ^^^^^^^
    """
    def __init__(
            self,
            name: str           = "multilayered-perceptron",
            activation: bool    = True
        ):
        """
        **Inputs:**

        * ``name: str``, the name of the MLP.
        * ``activation: bool``, whether to add relu activations or not.
        """
        raise NotImplementedError
    
    
    ## Evaluation
    def adhoc_loss(
            self,
            X: np.ndarray = None,
            Y: np.ndarray = None, 
            verbose: int = 0
        ) -> float:
        """
        **Dimensions:**
        
        * ``n``, the number of samples.
        * ``in_dim``, the MLP's input dimention (shape)

        **Inputs:**
        
        * ``X``, tensor with shape ``(n, in_dim)``.
        * ``Y``,tensor with shape ``(n,)``.
        
        ``Y`` should be a vector of the integer labels.

        **Outputs:**

        The loss measured on the given input.
        """
        raise NotImplementedError


    def adhoc_accuracy(
            self,
            X: np.ndarray = None,
            Y: np.ndarray = None, 
            verbose: int = 0
        ) -> float:
        """
        **Dimensions:**
        
        * ``n``, the number of samples.
        * ``in_dim``, the MLP's input dimention (shape)

        **Inputs:**
        
        * ``X``, tensor with shape ``(n, in_dim)``.
        * ``Y``,tensor with shape ``(n,)``.
        
        ``Y`` should be a vector of the integer labels.

        **Outputs:**

        The accuracy measured on the given input.
        """
        raise NotImplementedError


    ## Prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        **Dimensions:**
        
        * ``n``, the number of samples.
        * ``in_dim``, the MLP's input dimention (shape)

        **Inputs:**
        
        * ``X``, tensor with shape ``(n, in_dim)``.
        * ``Y``,tensor with shape ``(n,)``.
        
        ``Y`` should be a vector of the integer labels.
        
        """
        raise NotImplementedError
    

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
        **Dimensions:**
        
        * ``n``, the number of samples.
        * ``in_dim``, the MLP's input dimention (shape)
        * ``out_dim`` is the MLP's output dimention (shape)

        **Inputs:**
        
        * ``X``, tensor with shape ``(n, in_dim)``.
        * ``Y``,tensor with shape ``(n,)``.
        
        ``Y`` should be a vector of the integer labels.
        
        **Outputs:**

        ``S``, a tensor with shape ``(n, out_dim)``.
        """
        raise NotImplementedError


    def report(self) -> None:
        """
        Extensive report to stdout.
        """
        raise NotImplementedError

    ## I/O
    def report2file(
            self,
            filepath:str    = None,
            overwrite:bool  = True
        ) -> None:
        """
        Extensive report to file.
        """
        raise NotImplementedError

    ## Export
    def export2onnx(
            self,
            onnx_path:str   = None,
            overwrite:bool  = False
        ) -> None:
        """
        Export the MLP as `Open Neural Network eXchange (ONNX) <https://github.com/onnx>`__
        format.

        **Input:**
        
        * ``onnx_path: str``, the path to save the ``.onnx`` file.
            
            * The path *must* have the suffix ``.onnx``.
            * Default value is ``None``.

        * ``overwrite: bool``, overwrites the ``.onnx`` even if exists.

            * Default value is ``False``.

        **Behavior:**

        * If ``onnx_path != None``, then the exported file is saved to the given path.
        * Otherwise, the exported file is saved to the path ``./<self.name>/<self.name>.onnx``.

        """
        raise NotImplementedError


    ############################
    # Weight & Biases Matrices #
    ############################

    ## Weights & Matrices
    def get_weights(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Returning a tuple of the form ``(Ws, bs)``, where:

        * ``Ws = [W_0, W_1, ..., W_L]`` is a list of the weight matrices, and ``W_i`` is the weights of the i-th layer.
        * ``bs = [b_0, b_1, ..., b_L]`` is a list of the bias vectors, and ``b_i`` is the bias of the i-th layer.

        **Notes:**

        The returned np.ndarrays are a *deep copy* of the originals. Namely,
        when modifying the returned matrices, this does *not* entail a
        modification of the MLP's weights.
        """
        raise NotImplementedError
    
    def set_weights(
            self,
            Weights: np.ndarray,
            biases: np.ndarray
        ):
        """
        Setting the weight and biases of the network, w.r.t. the given values.
        """
        raise NotImplementedError

    ## Exports
    def export2csv(
            self,
            csv_dir:    str   = None,
            overwrite:  bool  = False,
            delim:      str   = default_delim,
        ) -> None:
        """
        Exports the networks weights as simple `.csv` files, in the
        designated directory. The files follow the format bellow:

        * ``W_<layer>.csv``, for the weight matrices.
        * ``b_<layer>.csv``, for the bias vectors.

        **Inputs:**

        * ``csv_dir: str``, the directory to save the csvs.
            
            * If the the directory doe not exists, it will be created.
            * Default value: ``None``.

        * ``overwrite: bool``, overwrite existing files.
            
            * If set to true all the parameter files, of the form ``<W|b>_<number>.csv`` will be erased.
            * Default value: ``False``.
        
        * ``delim: str``, the delimeter to be used in the csvs.
            
            * Default value: ``default_delim``, the latter constant is set to a single space.

        **Naming Convention:**

        *  The ``<layer>`` id will be *always* of length ``log(num_layers)``.

            * E.g., for a MLP of ``10`` layers and the 3rd weight matrix, we will have ``W_03.csv``.

        * If ``csv_dir == None``, then the csvs will be located under ``./<self.name>/``.
        * It also supports directory hierarchy. If ``csv_dir`` is of the form ``./<dir1>/<dir2>/ ... /<dirn>`` all the indermediate dirs will be created.

        """
        raise NotImplementedError
    
    def export2npz(
            self,
            filename:    str   = None,
            overwrite:  bool  = False
        ) -> None:
        """
        Convert to ``.npz`` format, a compresed collection of ``.npy`` files.
        ``.npy`` are binaries, supported by NumPy to store and retrieve matrices.
        There, we store the weight and bias matrices.
        """
        raise NotImplementedError
    
