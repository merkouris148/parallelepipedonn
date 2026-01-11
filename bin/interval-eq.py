# 3rd party libraries
import argparse
import numpy as np

# Guarantees
import sys
sys.path.append("..")
import geometry.interval as interval

## Constants
eq_all_ok   = 0
eq_neq      = 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-lb1", type=str, required=True, help="lower bound 1")
    required.add_argument("-ub1", type=str, required=True, help="upper bound 1")
    required.add_argument("-lb2", type=str, required=True, help="lower bound 2")
    required.add_argument("-ub2", type=str, required=True, help="upper bound 2")

    args = parser.parse_args()
    
    lb1 = np.genfromtxt(args.lb1, delimiter=" ")
    ub1 = np.genfromtxt(args.ub1, delimiter=" ")
    lb2 = np.genfromtxt(args.lb2, delimiter=" ")
    ub2 = np.genfromtxt(args.ub2, delimiter=" ")
    I1 = interval.Interval(lb1, ub1)
    I2 = interval.Interval(lb2, ub2)

    if I1 == I2:    exit(eq_all_ok)
    else:           exit(eq_neq)