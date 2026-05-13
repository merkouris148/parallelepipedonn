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

class CIFAR10(mlp.FromArchitecture):
    ## Constructor
    def __init__(
            self,
            architecture    = [3072, 1024, 512, 256, 128, 64, 32, 10],
            num_epochs      = 20,
            batch_size      = 512,
            name            = "CIFAR10-mlp"
        ):
        ## parameters for the dataset
        input_shape = (32, 32, 3)
        
        ## init super class
        super().__init__(architecture, input_shape, name)

        ## Load dataset
        CIFAR10_dataset = tf.keras.datasets.cifar10
        (
            self.X_train,
            self.Y_train
        ),(
            self.X_test,
            self.Y_test
        ) = CIFAR10_dataset.load_data()

        ## Preprocessing
        # self.X_train    = self.X_train / 255.0
        # self.X_test     = self.X_test  / 255.0

        ## fit
        self.fit(
            self.X_train,
            self.Y_train,
            epochs              = num_epochs,
            batch_size          = batch_size,
            validation_split    = 0.2,
            verbose = 1
        )



class CIFAR10FromDir(mlp.FromDir):
    def __init__(self, dir_path, delim=mlp.default_delim):
         ## parameters for the dataset
        input_shape = (32, 32, 3)

        super().__init__(dir_path, input_shape, delim)

        ## Load dataset
        CIFAR10_dataset = tf.keras.datasets.cifar10
        (
            self.X_train,
            self.Y_train
        ),(
            self.X_test,
            self.Y_test
        ) = CIFAR10_dataset.load_data()

        ## Preprocessing
        self.X_train    = self.X_train / 255.0
        self.X_test     = self.X_test  / 255.0