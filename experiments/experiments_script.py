###########################################################
# experiments_script.py
# --------------------------------------------------------
# Runs the ../parallelepipedonn.py for all the images of
# a specific directory *in parallel*. Reports the results
# in terms of Iterations, CPU time, Description Complexity,
# and the guarantees Min. Edge Length. For all these
# metrics we report the max, min, avg and variance values.
#
# Calls the experiments.Experiments class.
#
# Input:
#   1. A path to the input data directory
#   2. A path to a neural network's .onnx description
#   3. Number of threads
#   4. Maximum Iterations
#   5. Method to be applied
#
# Output:
#   If the input data directory path has the form:
#       +<path>/inputs
#   and <algo> is the given method, then in the folder
#       <path>/outputs/<algo>
#   will be included the following files:
#       a) errors.log
#       b) results.log
#       c) results.txt
###########################################################

import os
import sys
sys.path.append("..")
import cli.args as args

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import experiments#.experiments as experiments

#data_input_path = "../data/twos/inputs"
#onnx_path       = "../nn_weights/mnist_description.onnx"
#num_threads     = 20
#max_it          = 10
#method          = application.pfx_cyclic_dichotomic_bottom_up

if __name__=="__main__":
    
    ##############################
    # Handling the CLI Arguments #
    ##############################
    assert len(sys.argv) == 6

    # Input Data Directory
    data_input_path = sys.argv[1]
    assert os.path.isdir(data_input_path)
    
    # Model Description .onnx
    onnx_path = sys.argv[2]
    assert os.path.isfile(onnx_path)
    assert os.path.splitext(onnx_path)[1] == ".onnx"

    # Number of Threads
    num_threads = int(sys.argv[3])
    assert num_threads > 0

    # Max. Num. Iterations
    max_it = int(sys.argv[4])
    assert max_it > 0

    # Method
    method = sys.argv[5]
    assert method in args.args_algo.keys(), str(method)


    
    ###########################
    # Running the Experiments #
    ###########################
    exps = experiments.Experiments(
        data_input_path,
        onnx_path,
        num_threads,
        method,
        max_it
    )

    exps.do_experiments()
    
    if len(exps.exit_codes) < num_threads:
        exps.calculate_statistics()
        exps.save_results()
        exps.print_results()
    
    if exps.exit_codes != []:
        print("Some subprocess collapsed!")
        print("Exit codes:", exps.exit_codes)
