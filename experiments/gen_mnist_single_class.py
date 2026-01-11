## Libraries

# python libraries
import sys
import os
import random


# 3rd party libraries
import numpy as np
import matplotlib.image as plt_im

# custom
sys.path.append('..')
from neural_network import MNISTNeuralNetwork
import utils


## Constants
# weights path
weights_path = "../nn_weights/my_mnist_nn.weights.h5"

# Parameters
class_identifier    = 7
num_samples         = 1
shuffle             = True
# files
data_set_name       = "seven"
path_header         = "../data"
inputs_dir          = "inputs"
outputs_dir         = "outputs"
preds_filename      = "predictions.txt"
# csv parameters
sample_sfx          = ".csv"
#delim               = " "
# png parameters
img_sample_sfx      = ".png"


if __name__=="__main__":
    mnist_nn = MNISTNeuralNetwork(weights_path)
    mnist_nn.initialization()

    ## indices
    num_test_set    = len(mnist_nn.X_test)
    test_set_inds   = list(range(num_test_set))


    ## shuffle if needed
    if shuffle: random.shuffle(test_set_inds)


    ## mkdir if not exists
    # dataset parent directory
    dataset_path = utils.make_dataset_dir_path(path_header, data_set_name)
    if not os.path.exists(dataset_path): os.mkdir(dataset_path)
    
    # outputs directory
    inputs_path = utils.make_inputs_dir_path(path_header, data_set_name, inputs_dir)
    if not os.path.exists(inputs_path): os.mkdir(inputs_path)

    # outputs directory
    outputs_path = utils.make_outputs_dir_path(path_header, data_set_name, outputs_dir)
    if not os.path.exists(outputs_path): os.mkdir(outputs_path)
    

    ## generate samples
    # predictions file
    predictions_path    = utils.make_predictions_path(path_header, data_set_name, inputs_dir, preds_filename)
    predictions_file    = open(predictions_path, "a")

    # sample_filename_pfx
    sample_filename_pfx = utils.make_sample_filename_pfx(path_header, data_set_name, inputs_dir, class_identifier)

    # num samples generated
    num_samples_generated = 0
    for i in range(num_test_set):

        x_star  = mnist_nn.X_test[test_set_inds[i]]
        c_star  = mnist_nn.predict_argmax(x_star)[0]

        # if not in the correct class continue
        if c_star != class_identifier: continue

        ## num samples generated
        num_samples_generated += 1

        # do not exceed the num of samples
        if num_samples_generated > num_samples: break


        ## save file as .csv
        sample_filename = sample_filename_pfx + str(num_samples_generated) + sample_sfx
        np.savetxt(sample_filename, x_star)

        ## save image
        img_sample_filename = sample_filename_pfx + str(num_samples_generated) + img_sample_sfx
        plt_im.imsave(img_sample_filename, x_star, cmap="Greys", vmin=0, vmax=1)

        ## append class to predictions.txt
        predictions_file.write(str(class_identifier) + "\n")
    

    ## close file
    predictions_file.close()    