## Python Libraries
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import unittest
import sys
sys.path.append("/home/merkouris/Έγγραφα/Διδακτορικό/Έρευνα/ParallelepipedoNN/master-versions/master-version-3.2.1.6/")
#os.chdir("/home/merkouris/Έγγραφα/Διδακτορικό/Έρευνα/ParallelepipedoNN/master-versions/master-version-3.2.1.6/")

import time as t

## 3rd Party libraries
import numpy as np

## Custom Libraries
import geometry.interval as interval


class IntervalTestOps(unittest.TestCase):
    def __init__(self, methodName = "Testing Ops"):
        """
            We test the `Intervals` class: Meet Ops.
        """
        super().__init__(methodName)

        ## Inputs
        self.shape = (2,)
        self.lb = np.array([-1.0, -1.0])
        self.ub = np.array([1.0, 1.0])
        
        self.I = interval.Interval(self.lb, self.ub)

    def test00_meet(self):
        N       = 1_000_000
        new_ub  = self.ub
        print("\nDiammeter:", self.I.diam())

        tic = t.time()
        for i in range(N):
            new_ub  -= (np.ones(self.shape) * 1/N)
            J       = interval.Interval(self.lb, new_ub)
            self.I.meet(J)
            if i % 100_000 == 0: print("Diammeter:", self.I.diam())
        toc = t.time()

        elapsed_time = toc - tic
        print("\n",N, "meet operations in", round(elapsed_time, 4), "(secs)")

    def test01_join(self):
        N       = 1_000_000
        new_ub  = self.ub
        print("\nDiammeter:", self.I.diam())

        tic = t.time()
        for i in range(N):
            new_ub  += (np.ones(self.shape) * 1/N)
            J       = interval.Interval(np.zeros(self.shape), new_ub)
            self.I.join(J)
            if i % 100_000 == 0: print("Diammeter:", self.I.diam())
        toc = t.time()

        elapsed_time = toc - tic
        print("\n",N, "join operations in", round(elapsed_time, 4), "(secs)")

if __name__ == "__main__":
    print(os.getcwd)
    unittest.main(verbosity=2)