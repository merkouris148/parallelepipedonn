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
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tf2onnx

import numpy as np

## Custom Libraries
import sys
sys.path.append("../..")
import geometry.interval as intervals


#############
# Constants #
#############

default_delim = " "


#########
# RegEx #
#########

file_name_pattern = re.compile("(W|b)_0*[0-9]*.csv")
is_parameter_input = lambda filename: file_name_pattern.match(filename) is not None

#########
# Utils #
#########

def clear_parameters(dir: str, regex:re.Pattern =file_name_pattern):
    for f in os.listdir(dir):
        if regex.match(f): os.remove(os.path.join(dir, f))


def decompose_pos_neg(A:np.ndarray):
    A_pos = (A + np.abs(A))/2
    A_neg = (A - np.abs(A))/2

    return A_pos, A_neg

def check_dims_compatability(
        Weights:        t.List[np.ndarray],
        biases:         t.List[np.ndarray],
        input_shape:    t.List[int]
    ) -> None:

    assert len(Weights) == len(biases)
    
    L = len(Weights)
    old_weight_col_dim = np.prod(np.array(input_shape))
    for l in range(L):
        if len(Weights[l].shape) != 2:
            print("Dims Compatability Error: The ", l,"-th weight matrix is not 2D")
            exit(1)
        
        if len(biases[l].shape) != 1:
            print("Dims Compatability Error: The ", l,"-th bias vector is not 1D")
            exit(1)

        new_weight_row_dim = Weights[l].shape[0]
        new_weight_col_dim = Weights[l].shape[1]

        if new_weight_row_dim != old_weight_col_dim:
            print("Dims Compatability Error: The ", l,"-th matrix dimensions are incompatible")
            exit(1)
        
        if new_weight_col_dim != biases[l].shape[0]:
            print("Dims Compatability Error: The ", l,"-th bias vector dimension is incompatible")
            exit(1)

        old_weight_col_dim = new_weight_col_dim

def get_architecture(Weights: t.List[np.ndarray]):
    architecture0 = [Weights[0].shape[0]]
    architecture1 = [W.shape[1] for W in Weights]

    return architecture0 + architecture1


###########
# Classes #
###########
class MLPBaseClass(tf.keras.Sequential):
    """
        The Base Class for Multilayered Perceptrons. This class models a specific\\
        class of NN of the form:
        ```
        [Flatten, input_shpae],
        [Dense, ReLU, dim_0],
        [Dense, ReLU, dim_1],
        ...
        [Dense, ReLU, dim_(L-1)],
        ```
        This defines a L-layered MLP of the form `Z:IR^(dim_0) --> IR^(d_(L-1))`.
        
        **Data Members:**
        * `architecture: List[int]`, a list containing the dimensions of each layer,\\
        i.e. `architecture = [dim_0, dim_1, ..., dim_(L-1)].
        * `num_layers: int`, the number of layers, i.e. `num_layers == len(architecture)`.
        * `input_shape: Tuple[int]`, the shape on the inputs *before* flattening.
        * `out_dim: int`, the output dimension, i.e. `out_dim == architecture[-1]`.
        * `W: List[ndarray]`, the list of weights.
        * `b: List[ndarray]`, the list of biases. Naturally, `len(W) == len(b) ==`\\
        `len(architecture)`.
        * `X_train, X_test: ndarray`, the training datapoints. It holds that,\\
        `X_tain.shape == (n, input_shape)`, where `n` is the number of training samples.\\
        Similarly, `X_test.shape == (m, input_shape)`, where `n` is the number of test\\
        samples. Finally, `Y_train.shape == (n,)` and `Y_test.shape == (m,)`.

        **Notes:**
        * *All* the above data members should be properly initialized *before* the use\\
        of the provided methods. Otherwise, the methods behavior is unkown.
    """

    ## Constructor
    def __init__(self, name: str = "multilayered-perceptron", activation = True):
        """
           This Constructor is only to be called from the inherited\\
           classes. 
        """
        ## Init super class
        super().__init__(name=name)

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
    
    
    ## Evaluation
    def adhoc_loss(self, X: np.ndarray = None, Y: np.ndarray = None,  verbose: int = 0) -> float:
        """
            **Inputs:**
            * `X`, a tensor with shape `(n, in_dim)`, where `n` is the number of samples\\
                and `in_dim` is the MLP's input dimention (shape).
            * `Y`, a tensor with shape `(n,)`, where `n` is the number of samples.\\
            Essentially, `Y` should be a vector of the integer labels.

            **Outputs:**
            * The loss measured on the given input.
        """
        ## Input Checks
        #assert X != None and Y != None or X == None and Y == None
        if X is not None and Y is not None:
            assert X.shape[0]   == Y.shape[0]
            assert X.shape[1:]  == self.in_shape


        ## Evaluation Data
        X = self.X_test if X is None else X
        Y = self.Y_test if Y is None else Y

        return round(self.evaluate(X, Y, verbose=verbose)[0], 4)



    def accuracy(self, X: np.ndarray = None, Y: np.ndarray = None,  verbose: int = 0) -> float:
        """
            **Inputs:**
            * `X`, a tensor with shape `(n, in_dim)`, where `n` is the number of samples\\
                and `in_dim` is the MLP's input dimention (shape).
            * `Y`, a tensor with shape `(n,)`, where `n` is the number of samples.\\
            Essentially, `Y` should be a vector of the integer labels.

            **Outputs:**
            * The accuracy measured on the given input.
        """
        ## Input Checks
        #assert X != None and Y != None or X == None and Y == None
        if X is not None and Y is not None:
            assert X.shape[0]   == Y.shape[0]
            assert X.shape[1:]  == self.in_shape
        

        ## Evaluation Data
        X = self.X_test if X is None else X
        Y = self.Y_test if Y is None else Y

        return round(self.evaluate(X, Y, verbose=verbose)[1], 4)




    ## Prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            **Inputs:**
            * `X`, a tensor with shape `(n, in_shape)`, where `n` is the number of samples\\
                and `in_shape` is the MLP's input dimention (shape).
            
            **Outputs:**
            * `Y`, a tensor with shape `(n,)`, where `n` is the number of samples.\\
            Essentially, `Y` should be a vector of the integer labels.

            **Notes:**
            * If `X` is of shape `in_shape`, the function will expand its dimension\\
            to match the shape of the tensor that TF expects.
        """
        ## Input Checks
        assert X is not None
        if len(X.shape) == len(self.in_shape): X = np.expand_dims(X, axis=0)

        return np.argmax(super().predict(X, verbose=0), axis=1)
    

    def scores(self, X: np.ndarray) -> np.ndarray:
        """
            **Inputs:**
            * `X`, a tensor with shape `(n, in_shape)`, where `n` is the number of samples\\
                and `in_shape` is the MLP's input dimention (shape).
            
            **Outputs:**
            * `Y`, a tensor with shape `(n, out_dim)`, where `n` is the number of samples\\
                and `out_dim` is the MLP's output dimention (shape).
            
            **Notes:**
            * If `X` is of shape `in_shape`, the function will expand its dimension\\
            to match the shape of the tensor that TF expects.
        """
        ## Input Checks
        assert X is not None
        if len(X.shape) == len(self.in_shape): X = np.expand_dims(X, axis=0)
        

        return super().predict(X, verbose=0)


    ## Report
    def __str__(self) -> str:
        init_method = "Training" if not self.from_file else "From File (" + self.filepath + ")"

        s =    "=================================================================" + "\n"
        s +=     "~~ Reporting ~~"                           + "\n"
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
        for layer in self.layers:
            s += f"{'':<4}{layer.name:<12}"
            s += f"{str(layer.input_shape[1:]):<17}"
            s += f"{str(layer.output_shape[1:]):<20}"
            if len(layer.get_weights()) == 0:   s += f"{str(layer.get_weights()):<30}" + "\n"
            else:                               s += f"{str(layer.get_weights()[0].shape):<30}" + "\n"
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
            filepath:str = None,
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
        """
            Export the MLP as [Open Neural Network eXchange (ONNX)](https://github.com/onnx) format.\\
            **Input:**
            * `onnx_path: str`, the path to save the `.onnx` file.\\
            The path **must** have the suffix `.onnx`. Default value\\
            is set to `None`.
            * `overwrite: bool`, overwrites the `.onnx` even if exists.\\
            Default value is set to `False`.

            **Behavior:**
            * If `onnx_path != None`, then the exported file is saved to\\
            the given path.
            * Otherwise, the exported file is saved to the path\\
            `./<self.name>/<self.name>.onnx`.

            **Notes:**
            * A wrapper of `tf2onnx.convert.from_keras()`, see [here](https://github.com/onnx/tensorflow-onnx).
        """

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
            tf2onnx.convert.from_keras(self, output_path = onnx_path)


    def export2csv(
            self,
            csv_dir:    str   = None,
            overwrite:  bool  = False,
            delim:      str   = default_delim,
        ) -> None:
        """
            Exports the networks weights as simple `.csv` files, in the\\
            designated directory. The files follow the format bellow:
            * `W_<layer>.csv`, for the weight matrices.
            * `b_<layer>.csv`, for the bias vectors.

            **Inputs:**
            * `csv_dir: str`, the directory to save the csvs. If the\\
            the directory doe not exists, it will be created.\\
            Default value: `None`.
            * `overwrite: bool`, overwrite existing files. If set to true\\
            all the parameter files, of the form `<W|b>_<number>.csv` will\\
            be erased.\\
            Default value: `False`.
            * `delim: str`, the delimeter to be used in the csvs.\\
            Default value: `default_delim`, the latter constant is\\
            set to a single space.

            **Naming Convention:**
            *  The `<layer>` id will be *always* of length `log(num_layers)`.\\
            Namely, for a MLP of `10` layers and the 3rd weight matrix,\\
            we will have `W_03.csv`.
            * If `csv_dir == None`, then the csvs will be located\\
            under `./<self.name>/`.
            * It also supports directory hierarchy. If `csv_dir` is of the\\
            form `./<dir1>/<dir2>/ ... /<dirn>` all the indermediate dirs\\
            will be created.
        """

        ## default csv dir
        csv_dir = "./" + self.name if csv_dir is None else csv_dir
        # if overwrite, clear previous parameters
        if os.path.isdir(csv_dir) and overwrite: clear_parameters(csv_dir)
        # create csv_dir, if not exists
        if not os.path.isdir(csv_dir): os.makedirs(csv_dir)

        ## export each parameter
        # we skip the 1st layer, since it is the "weightless" flatten layer
        for i in range(1, self.num_hid_layers+1):
            pfx_zeros   = ((self.num_hid_layers - i)//10) * "0"

            output_path = csv_dir + "/" + "W" + "_" + pfx_zeros + str(i) + ".csv"
            if not os.path.isfile(output_path) or overwrite:
                np.savetxt(output_path, self.layers[i].get_weights()[0], delimiter=delim)

            output_path = csv_dir + "/" + "b" + "_" + pfx_zeros + str(i) + ".csv"
            if not os.path.isfile(output_path) or overwrite:
                np.savetxt(output_path, self.layers[i].get_weights()[1], delimiter=delim)


    ## Operations
    def semantic_eq(self, mlp) -> bool:
        if self.num_hid_layers != mlp.num_hid_layers: return False

        for k in range(1, self.num_hid_layers + 1):
            if (
                self.layers[k].get_weights()[0] !=\
                mlp.layers[k].get_weights()[0]
            ).any()\
            or\
            (
                self.layers[k].get_weights()[1] !=\
                mlp.layers[k].get_weights()[1]
            ).any():
            
                return False
        
        return True


    ## Mutators
    def update_parameters(self, Weights, biases):
        assert len(self.weights) == len(Weights) + len(biases)

        ## For each layer
        for k in range(1, self.num_hid_layers+1):
            self.layers[k].set_weights([Weights[k-1], biases[k-1]]) 


    def build(self):
        tf_input_shape = tuple([None] + list(self.in_shape))
        super().build(tf_input_shape)


    ## Bound Propagation
    def propagate_bounds(
            self,
            domain: intervals.Interval
        ) -> t.Tuple[
            t.List[intervals.Interval],
            t.List[intervals.Interval]
        ]:
        
        """
            The simple bound propagation algorithm, appeared in:
            [Li et al., (IEEE TDSC 2022)], [Liu et al., (ICML 2019)] and
            [Gowal et al., (2018)]
        """
        ## Preprocessing 
        lb      = domain.lb.flatten()
        ub      = domain.ub.flatten()
        I_hat   = []
        I       = []
        I_hat.append(intervals.Interval(lb, ub))

        ## Propagate
        for layer in self.layers:
            if layer.name == "flatten": continue

            # get layer parameters
            W = layer.get_weights()[0]
            b = layer.get_weights()[1]

            # decompose to positive and negative
            W_pos, W_neg = decompose_pos_neg(W)

            ## propagate bound through the linear part
            lb  = np.matmul(I_hat[-1].lb, W_pos) + np.matmul(I_hat[-1].ub, W_neg) + b
            ub  = np.matmul(I_hat[-1].ub, W_pos) + np.matmul(I_hat[-1].lb, W_neg) + b
            I.append(intervals.Interval(lb, ub))

            ## propagate bound through ReLU
            lb = np.maximum(np.zeros(lb.shape), I[-1].lb)
            ub = np.maximum(np.zeros(ub.shape), I[-1].ub)
            I_hat.append(intervals.Interval(lb, ub))

        
        return I_hat, I

    ## Utilities
    def random_points(
            self,
            num_of_rand_inputs: int = 1000,
            seed: int = 0, type: str = "test"
        ) -> np.ndarray:

        assert type == "test" or type == "train"

        N           = self.X_test.shape[0]
        rnjeasus    = np.random.default_rng(seed)
        indices     = rnjeasus.integers(0, N, num_of_rand_inputs)

        if type == "test":  return self.X_test[indices, :, :]
        if type == "train": return self.X_train[indices, :, :]



class FromArchitecture(MLPBaseClass):
    ## Constructor
    def __init__(
                    self,
                    architecture:   t.List[int],
                    input_shape:    t.Tuple[int],
                    name:           str             = "multilayered-perceptron",
                    activation:     bool            = True
                ):
        """
            Initialize a MLP using architecture parameters, i.e.\\
            1. `architecture: List[int]`, a list of dimentions.
            2. `input_shape: Tuple[int]`, the shape on the inputs before flattening.
            3. `name: str`, the name of the MLP.\\
                Default Value: `multilayered-perceptron`.
            4. `activation: bool`, a flag to either enable or disable the `relu`
                activations.
                Default Value: `True`.
        """

        ## Init super class
        super().__init__(name, activation)

        ## Architecture
        self.architecture   = architecture
        self.num_hid_layers = len(architecture) - 1 # number of hidden layers
        self.in_shape       = input_shape
        self.out_dim        = self.architecture[-1]
        
        assert (np.array(self.in_shape) > 0).all()
        assert (np.array(self.architecture) > 0).all()
        assert np.prod(np.array(self.in_shape)) == self.architecture[0], self.architecture[0]


        ## First layer
        self.add(
            tf.keras.layers.Flatten(
                name = "flatten"
                #input_shape=self.in_shape
            )
        )

        ## Create the layers
        ind     = 0
        activ   = 'relu' if activation else None
        for l in self.architecture[1:]:
            self.add(
                tf.keras.layers.Dense(
                    l,
                    name                = "dense_" + str(ind),
                    kernel_regularizer  = tf.keras.regularizers.L2(0.01),
                    bias_regularizer    = tf.keras.regularizers.L2(0.01),
                    activation          = activ
                )
            )
            ind += 1

        # Compile the model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3),                              # ADAM optimizer
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # Loss Function
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],  # Accuracy metric
        )
        self.build()




class FromWeights(FromArchitecture):
    def __init__(
            self,
            Weights:        t.List[np.ndarray],
            biases:         t.List[np.ndarray],
            input_shape:    t.Tuple[int],
            name:           str             = "multilayered-perceptron",
            activation:     bool            = True
    ):
        """
            Initialize a MLP from a list of weights and biases.

            **Inputs:**
            1. `Weights: List[np.ndarray]`, a list of weights.
            2. `biases: List[np.ndarray]`, a list of biases.
            3. `input_shape: Tuple[int]`, the shape on the inputs before flattening.
            4. `name: str`, the name of the MLP.\\
                Default Value: `multilayered-perceptron`.
            5. `activation: bool`, a flag to either enable or disable the `relu`
                activations.
                Default Value: `True`.
        """

        assert len(Weights) == len(biases)
        ## Check the dimensions compatability
        check_dims_compatability(Weights, biases, input_shape)
        ## Get architecture
        architecture = get_architecture(Weights)

        ## Init. super-class
        super().__init__(architecture, input_shape, name, activation)

        self.update_parameters(Weights, biases)

        



class FromDir(FromWeights):
    def __init__(
            self,
            dir_path:       str,
            input_shape:    t.Tuple[int],
            delim:          str             = default_delim,
            activation:     bool            = True
    ):
        """
            Initialize a MLP from a dir of csv files:

            **Inputs:**\\
            1. `dir_path: str`, the path to the diractory.
            2. `input_shape: Tuple[int]`, the shape on the inputs before flattening.
            3. `delim: str`, the delimiter for the `.csv` files.
            4. `activation: bool`, a flag to either enable or disable the `relu`
                activations.
                Default Value: `True`.
        """

        ## Weights csvs
        W_csvs = [
            dir_path + "/" + item
            for item in os.listdir(dir_path)
            if
                os.path.isfile(dir_path + "/" + item)
            and is_parameter_input(item)
            and item[0] == "W"
        ]
        W_csvs.sort()

        ## Bias csvs
        b_csvs = [
            dir_path + "/" + item
            for item in os.listdir(dir_path)
            if
                os.path.isfile(dir_path + "/" + item)
            and is_parameter_input(item)
            and item[0] == "b"
        ]
        b_csvs.sort()

        ## non empty parameters
        assert W_csvs != [], "No weight matrices in this dir!"
        assert b_csvs != [], "No bias vectors in this dir!"

        ## Equal number of weight matrices and biases
        assert len(W_csvs) == len(b_csvs), "len(W_csvs) != len(b_csvs)"
        
        ## Load the Weights and biases
        matfromcsv = lambda file: np.genfromtxt(file, delimiter=delim)
        Weights = list(map(matfromcsv, W_csvs))
        biases  = list(map(matfromcsv, b_csvs))

        # name
        name = dir_path.split("/")[-1]

        ## Init super class
        super().__init__(Weights, biases, input_shape, name, activation)
        
        ## From file
        self.from_file  = True
        self.filepath   = dir_path
    


class FromRandom(FromWeights):
    def __init__(
        self,
        architecture:   t.List[int],
        input_shape:    t.Tuple[int],
        seed:           int             = 0,
        name:           str             = "multilayered-perceptron",
        activation:     bool            = True
    ):
        """
            Creating a MLP from random weight matrices.
            **Inputs:**
            1. `architecture: List[int]`, a list of each layer output dimensions.
            2. `input_shape: Tuple[int]`, the input shape.
            3. `seed: int`, the seed for the random number generator.\\
            Default Value: `seed = 0`.
            4. `name: str`, the name of the MLP.\\
            Default Value: `name = "multilayered-perceptron"`.
            5. `activation: bool`, a flag to either enable or disable the `relu`
                activations.
                Default Value: `True`.
            
            **Notes:**
            * The weights and biases take values in `[-1, 1]`, w.r.t. their shape.
        """
        
        num_layers  = len(architecture) - 1 # number of hidden layers
        row_dim     = architecture[0]
        Ws          = []
        bs          = []
        
        for k in range(1, num_layers+1):
            col_dim         = architecture[k]

            ## Create Weight Matrix
            weight_dom_lb   = -1.0 * np.ones((row_dim, col_dim))
            weight_dom_ub   = 1.0 * np.ones((row_dim, col_dim))
            WeightDom       = intervals.Interval(
                                weight_dom_lb,
                                weight_dom_ub
                            )
            W               = WeightDom.random_points(1, seed)[0]
            Ws.append(W)

            ## Create Bias Vector
            bias_dom_lb   = -1.0 * np.ones((col_dim,))
            bias_dom_ub   = 1.0 * np.ones((col_dim,))
            BiasDom       = intervals.Interval(
                                bias_dom_lb,
                                bias_dom_ub
                            )
            b             = BiasDom.random_points(1, seed)[0]
            bs.append(b)

            ## update col dim
            row_dim       = col_dim


        super().__init__(Ws, bs, input_shape, name, activation)