import os
import unittest
import numpy as np
import datetime
from hymod import mutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hymod)')
        self.dt = None # TODO

    def test_turc_mezentsev(self):
        P = np.linspace(700, 1200, 20)
        PE = np.linspace(1000, 2000, 20)
        n = 2.3
        Q1 = mutils.turc_mezentsev(n, P, PE)
        Q2 = P*(1-1/(1+(P/PE)**n)**(1/n))
        self.assertTrue(all(Q1==Q2))

    def test_memusage(self):
        mem = mutils.getmemusage()


if __name__ == "__main__":
    unittest.main()
