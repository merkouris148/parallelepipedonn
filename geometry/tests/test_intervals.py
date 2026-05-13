import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import sys
sys.path.append("..")

import numpy as np

import interval

class IntervalTest(unittest.TestCase):
    def __init__(self, methodName = "Testing Intervals"):
        """
            We test the `Intervals` class.
        """
        super().__init__(methodName)

        ## Inputs
        self.shape = (2, 2)
        self.x = np.array(
                    [[-1, -1],
                     [-1, -1]]
                )
        
        self.y = np.array(
                    [[1, 1],
                     [1, 1]]
                )
        
        self.I = interval.Interval(self.x, self.y)


    def test00_report(self):
        self.I.report()
    
    def test01_empty_n_sigleton(self):
        I1 = interval.Interval(self.y, self.x)
        self.assertTrue(I1.empty())

        I2 = interval.Interval(self.x, self.x)
        self.assertFalse(I2.empty())

        I3 = interval.Interval(self.x, self.x)
        self.assertTrue(I3.sigleton())

        I4 = interval.Interval(self.x, self.y)
        self.assertFalse(I4.sigleton())
        self.assertFalse(I4.empty())

    def test02_contains(self):
        x = np.zeros(self.shape)
        self.assertTrue(x in self.I)
    
    def test03_eq(self):
        I_prime = interval.Interval(self.x, self.y)
        self.assertTrue(self.I  == I_prime)
        self.assertFalse(self.I != I_prime)

    def test04_ne(self):
        I_prime = interval.Interval(self.x, self.x)
        self.assertTrue(self.I  != I_prime)
        self.assertFalse(self.I == I_prime)

    def test05_lt(self):
        I_sub = interval.Interval(0.5 * self.x, 0.5 * self.y)
        I_sup = interval.Interval(2.0 * self.x, 2.0 * self.y)

        self.assertTrue(self.I  < I_sup)
        self.assertTrue(I_sub   < self.I)

    def test06_le(self):
        I_sub   = interval.Interval(0.5 * self.x, 0.5 * self.y)
        I_sup   = interval.Interval(2.0 * self.x, 2.0 * self.y)
        I_prime = interval.Interval(self.x, self.y)

        self.assertTrue(self.I  <= I_sup)
        self.assertTrue(self.I  <= I_prime)
        self.assertTrue(I_sub   <= self.I)
        self.assertTrue(I_prime <= self.I)

    def test07_gt(self):
        I_sub = interval.Interval(0.5 * self.x, 0.5 * self.y)
        I_sup = interval.Interval(2.0 * self.x, 2.0 * self.y)

        self.assertTrue(I_sup   > self.I)
        self.assertTrue(self.I  > I_sub)

    def test08_ge(self):
        I_sub   = interval.Interval(0.5 * self.x, 0.5 * self.y)
        I_sup   = interval.Interval(2.0 * self.x, 2.0 * self.y)
        I_prime = interval.Interval(self.x, self.y)

        self.assertTrue(I_sup   >= self.I)
        self.assertTrue(I_prime >= self.I)
        self.assertTrue(self.I  >= I_sub)
        self.assertTrue(self.I  >= I_prime)
    
    def test09_or(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -0.1 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )

        self.assertTrue(I1 or I2 == self.I)

    def test10_and(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -0.1 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )

        self.assertTrue((I1 and I2).empty())
    
    def test11_join(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -0.1 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )
        I_or = I1 | I2

        I1.join(I2)
        self.assertTrue(I1 == I_or)

    def test12_meet(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -0.1 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )
        I_and = I1 & I2

        I1.meet(I2)
        self.assertTrue(I1 == I_and)
    
    def test13_add(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -1.0 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )
        I_add_1 = I1 + I2

        I_add_2 = interval.Interval(
            -0.5 * np.ones(self.shape),
            0.5 * np.ones(self.shape),
        )

        self.assertTrue(I_add_1 == I_add_2)

    def test14_sub(self):
        I1 = interval.Interval(
            0.5 * np.ones(self.shape),
            np.ones(self.shape)
        )

        I2 = interval.Interval(
            -1.0 * np.ones(self.shape),
            -0.5 * np.ones(self.shape)
        )
        I_sub_1 = I1 - I2

        I_sub_2 = interval.Interval(
            1.5 * np.ones(self.shape),
            1.5 * np.ones(self.shape),
        )

        self.assertTrue(I_sub_1 == I_sub_2)
    
    def test15_vol_percision(self):
        x = np.zeros((100))
        y = 0.1 * np.ones((100))

        I = interval.Interval(x, y)
        I.report()

if __name__ == "__main__":
    unittest.main(verbosity=2)