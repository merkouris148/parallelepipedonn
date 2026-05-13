#############
# Libraries #
#############

## 3rd Party Libraries
# TensorFlow
import tensorflow as tf

# Custom libraris
import perceptrons.multilayer as mlp

###########
# Classes #
###########

class Fashion(mlp.FromArchitecture):
    ## Constructor
    def __init__(
            self,
            architecture    = [784, 64, 10],
            num_epochs      = 6,
            batch_size      = 128,
            name            = "Fashion-MNIST-mlp"
        ):
        ## parameters for the dataset
        input_shape = (28, 28)
        
        ## init super class
        super().__init__(architecture, input_shape, name)

        ## Load dataset
        Fashion_dataset = tf.keras.datasets.fashion_mnist
        (
            self.X_train,
            self.Y_train
        ),(
            self.X_test,
            self.Y_test
        ) = Fashion_dataset.load_data()

        ## Preprocessing
        self.X_train    = self.X_train / 255.0
        self.X_test     = self.X_test  / 255.0

        ## fit
        self.fit(
            self.X_train,
            self.Y_train,
            epochs              = num_epochs,
            batch_size          = batch_size,
            validation_split    = 0.2,
            verbose             = 1
        )



class FashionFromDir(mlp.FromDir):
    def __init__(self, dir_path, delim=mlp.default_delim):
        ## parameters for the dataset
        input_shape = (28, 28)

        super().__init__(dir_path, input_shape, delim)

        ## Load dataset
        Fashion_dataset = tf.keras.datasets.fashion_mnist
        (
            self.X_train,
            self.Y_train
        ),(
            self.X_test,
            self.Y_test
        ) = Fashion_dataset.load_data()

        ## Preprocessing
        self.X_train    = self.X_train / 255.0
        self.X_test     = self.X_test  / 255.0