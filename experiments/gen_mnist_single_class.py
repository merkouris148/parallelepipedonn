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
from neural_network import HandSignNeuralNetwork


## Constants
# weights path
weights_path = "../nn_weights/mnist_nn-32.h5"

# Parameters
class_identifier    = 7     # change here to generate instances of a different class
num_samples         = 5     # change here to generate a different number of instances
shuffle             = True
# files
path_header         = "../data"
data_set_name       = "/inputs"
outputs_dir         = "/MNIST/"
# outputs_dir         = "/FashionMNIST/" # (uncommend to generate FashionMNIST instances)

preds_filename      = "predictions.txt"

# csv parameters
sample_sfx          = ".csv"
#delim               = " "
# png parameters
img_sample_sfx      = ".png"


if __name__=="__main__":
    neural_net = MNISTNeuralNetwork(weights_path)
    # neural_net = FashionMNISTNeuralNetwork(weights_path) # (uncommend to generate FashionMNIST instances)
    neural_net.initialization()

    ## indices
    num_test_set    = len(neural_net.X_test)
    test_set_inds   = list(range(num_test_set))


    ## shuffle if needed
    if shuffle: random.shuffle(test_set_inds)


    ## mkdir if not exists
    # outputs directory
    outputs_path = path_header + data_set_name + outputs_dir
    if not os.path.exists(outputs_path): os.mkdir(outputs_path)
    

    ## generate samples
    # predictions file
    predictions_path    = outputs_path + preds_filename
    predictions_file    = open(predictions_path, "a")

    # sample_filename_pfx
    sample_filename_pfx =  outputs_path + str(class_identifier) +"_"

    # num samples generated
    num_samples_generated = 1
    for i in range(num_test_set):

        x_star  = neural_net.X_test[test_set_inds[i]].reshape(28, 28)
        c_star  = neural_net.predict_argmax(x_star)[0]

        # if not in the correct class continue
        if c_star != class_identifier:          continue
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



        ## num samples generated
        num_samples_generated += 1
    

    ## close file
    predictions_file.close()    
