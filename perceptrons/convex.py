#############
# Libraries #
#############

## Python Libraries
## typing
import typing as t

## 3rd Party Libraries
import tensorflow as tf
import numpy as np

## Custom Libraries
import sys
sys.path.append("..")
import geometry.interval as intervals
import perceptrons.multilayer as mlp

###########
# Classes #
###########

class ConvexApprox(mlp.FromWeights):

    def __init__(
            self,
            relu_model: mlp.MLPBaseClass,
            domain:     intervals.Interval
        ):
        """
            Constructing the Tight Convex Approximation of a MLP.
            
            **Inputs:**
            * `relu_model: mlp.MLPBaseClass`, the ReLU model to be\\
            approximated.
            * `domain: intervals.Interval`, the interval w.r.t. which\\
            the approximation will be constructed.
        """

        ## Resolve name
        conv_name = relu_model.name + "-conv-approx"

 
        ## Propagate the Domain
        self.preactivation_bounds = relu_model.propagate_bounds(domain)[1]


        ## Normalize Intervals
        for I in self.preactivation_bounds:
            if not I.empty() and (I.ub < np.zeros(I.shape)).any():
                I.ub = np.maximum(I.ub, np.zeros(I.shape))

            if not I.empty() and (I.lb > np.zeros(I.shape)).any():
                I.lb = np.minimum(I.lb, np.zeros(I.shape))


        ## Create the Convex Activation Layers
        ConvexActivation_Ws = []
        ConvexActivation_bs = []
        for I in self.preactivation_bounds:
            v = I.ub / (I.ub - I.lb)
            b = -1.0 * v * I.lb
            W = np.diag(v)

            ConvexActivation_Ws.append(W)
            ConvexActivation_bs.append(b)
        

        ## Create Weights and Biases
        ind     = 0
        Weights = []
        biases  = []
        for layer in relu_model.layers:
            if layer.name == "flatten": continue

            W_affine = layer.get_weights()[0]
            Weights.append(W_affine)
            Weights.append(ConvexActivation_Ws[ind])

            b_affine = layer.get_weights()[1]
            biases.append(b_affine)
            biases.append(ConvexActivation_bs[ind])

            ind += 1
        

        ## initialize superclass
        super().__init__(
            Weights,
            biases,
            relu_model.in_shape,
            conv_name,
            False
        )

        ## Input
        self.X_train    = relu_model.X_train
        self.Y_train    = relu_model.Y_train
        self.X_test     = relu_model.X_test
        self.Y_test     = relu_model.Y_test