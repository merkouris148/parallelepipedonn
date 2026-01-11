# a neural network wrapper
# We create a simple NN to explain locally.
# We use a standard TF example (slightly modified):
# https://www.tensorflow.org/tutorials/keras/classification

#############
# Libraries #
#############

## Python Libraries
#from pprint import pprint
import os.path

## 3rd Party Libraries
# TensorFlow
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tf2onnx

import numpy as np


#############
# Constants #
#############

#checkpoint_path = "./nn/mnist-nn.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

messages = False


####################
# Helper Functions #
####################

def file_exists(path):
    return os.path.isfile(path)


class MNISTNeuralNetwork:

    ##################
    # Initialization #
    ##################

    def __init__(self, weights_path):
        
        ## Data
        self.X_train    = None
        self.Y_train    = None
        self.X_test     = None
        self.Y_test     = None

        ## Model
        self.model = None

        ## Callback Checkpoint
        self.weights_path   = weights_path
        self.weights_dir    = os.path.dirname(self.weights_path)

        ## Parameters
        self.accuracy   = None

        ## Book-keeping
        self.where_weights = None

        ## Sate
        #self.checkpoint_created = False
        self.data_loaded        = False
        self.data_preprocessed  = False
        self.model_created      = False
        self.weights_filled     = False
    

    def initialization(self):
        ## Preconsition
        #assert (self.data_loaded or self.data_preprocessed or\
        #        self.model_created or self.weights_filled) == False
        assert self.blank_instance()

        ## Data
        self._load_MNIST_data()
        self._preprocessing()

        ## Create Model Architecture
        self._create_model()

        if file_exists(self.weights_path):
            self._weights_from_file()
            self.where_weights = "file"
            if messages: print("Msg: Loading weights from: " + self.weights_path)
        else:
            self._weights_from_data()
            self.where_weights = "data"
            if messages: print("Msg: Learning weights from data.")

        ## Postcondition
        #assert (self.data_loaded and self.data_preprocessed and\
        #        self.model_created and self.weights_filled) == True
        assert self.init_completed()
    

    ##############
    # Predicates #
    ##############

    def init_completed(self):
        return  self.data_loaded        and\
                self.data_preprocessed  and\
                self.model_created      and\
                self.weights_filled

    def blank_instance(self):
        return (
                    self.data_loaded        or\
                    self.data_preprocessed  or\
                    self.model_created      or\
                    self.weights_filled
                ) == False

    #############
    # Accessors #
    #############

    def get_accuracy(self):
        #assert (self.data_loaded and self.data_preprocessed and\
        #        self.model_created and self.weights_filled) == True
        assert self.init_completed()

        return self.accuracy


    ########
    # Data #
    ########

    def _load_MNIST_data(self):
        ## Precondition
        assert self.data_loaded == False

        mnist = tf.keras.datasets.mnist
        (
            self.X_train,
            self.Y_train
        ),(
            self.X_test,
            self.Y_test
        ) = mnist.load_data()


        ## Postcondition
        self.data_loaded = True
    

    def _preprocessing(self):
        # ## Precondition
        assert self.data_loaded         == True
        assert self.data_preprocessed   == False

        self.X_train    = self.X_train / 255.0
        self.X_test     = self.X_test  / 255.0

        # ## Postcondition
        self.data_preprocessed = True
    


    #########
    # Model #
    #########

    def _create_model(self):
        ## Precondition
        assert self.data_preprocessed   == True
        assert self.model_created       == False

        ## Create & Train the Model ##
        # Define the model
        self.model = tf.keras.models.Sequential([
                                        # Flatten input to vector
                                        tf.keras.layers.Flatten(
                                            input_shape=(28, 28)
                                        ),
                                        
                                        # Fully connected Linear Layers
                                        # ReLU activation functions
                                        tf.keras.layers.Dense(
                                            32,
                                            #kernel_initializer  = 'ones',
                                            #bias_initializer    = 'zeros',
                                            # (default) kernel_initializer='glorot_uniform'
                                            kernel_regularizer  = tf.keras.regularizers.L2(0.01),
                                            bias_regularizer    = tf.keras.regularizers.L2(0.01),
                                            activation          = 'relu'
                                        ),

                                        # Fully connected Linear Layers
                                        tf.keras.layers.Dense(10)
                                    ])
        # Compile the model
        self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(0.001),                              # ADAM optimizer
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # Loss Function
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],  # Accuracy metric
                    )

        ## Postcondition
        self.model_created = True



    def _weights_from_data(self):
        ## Precondition
        assert self.model_created   == True
        assert self.weights_filled  == False


        # Training
        self.model.fit(
                        self.X_train,
                        self.Y_train,
                        epochs=6,
                        verbose = 1
                    )
        #self.accuracy = self.evaluate()

        # Saving Weights
        self.model.save_weights(self.weights_path)

        # Recalculating Accuracy
        self.accuracy = self.evaluate()

        ## Postcondition
        self.weights_filled = True
    

    def _weights_from_file(self):
        ## Precondition
        assert self.model_created   == True
        assert self.weights_filled  == False


        # Saving Weights
        self.model.load_weights(self.weights_path)

        # Recalculating Accuracy
        self.accuracy = self.evaluate()

        ## Postcondition
        self.weights_filled = True


    ##############
    # Evaluation #
    ##############
    def evaluate(self, verbose=0):
        return self.evaluate_test(verbose)

    def evaluate_test(self, verbose=0):
        return round(self.model.evaluate(self.X_test, self.Y_test, verbose=verbose)[1], 4)
    
    def evaluate_train(self, verbose=0):
        return round(self.model.evaluate(self.X_train, self.Y_train, verbose=verbose)[1], 4)

    def predict(self, X):
        assert self.init_completed()

        prediction = self.model(X.reshape([1, 28, 28]))
        return prediction.numpy()
    
    def predict_argmax(self, X):
        assert self.init_completed()

        predictions_vector  = self.predict(X)[0]
        prediction_class    = np.argmax(predictions_vector)
        prediction_value    = predictions_vector[prediction_class]
        return prediction_class, prediction_value
    
    ##########
    # Export #
    ##########

    def export2onnx(self, onnx_path):
        assert self.init_completed()

        if not file_exists(onnx_path):
            return tf2onnx.convert.from_keras(self.model, output_path = onnx_path)