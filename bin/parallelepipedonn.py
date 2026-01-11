# python libraries
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# custom libraries
import cli.runner as runner


if __name__=="__main__":
    run_app = runner.Runner(sys.argv)
    run_app.exec()