# 3rd party libraries
import argparse
import numpy as np

# Guarantees
import sys
sys.path.append("..")
import guarantees.parallelepipedal as guarantees

## Constants
memebr_ok   = 0
member_nok  = 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-x", type=str, required=True, help="input point")
    required.add_argument("-lb", type=str, required=True, help="lower bound")
    required.add_argument("-ub", type=str, required=True, help="upper bound")

    args = parser.parse_args()
    
    x = np.genfromtxt(args.x, delimiter=" ")
    g = guarantees.ParallelepipedalGuarantee(args.lb, args.ub)

    if x in g:  exit(memebr_ok)
    else:       exit(member_nok)