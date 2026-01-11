# 3rd party libraries
import numpy as np
import argparse


## Constants
leq   = 0
nleq  = 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-x", type=str, required=True, help="left hand side of x <= y")
    required.add_argument("-y", type=str, required=True, help="right hand side of x<= y")

    args = parser.parse_args()
    
    x = np.genfromtxt(args.x, delimiter=" ")
    y = np.genfromtxt(args.y, delimiter=" ")


    if (x <= y).all():  exit(leq)
    else:               exit(nleq)