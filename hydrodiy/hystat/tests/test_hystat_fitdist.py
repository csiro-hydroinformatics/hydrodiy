import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hystat import fitdist

class FitdistTestCase(unittest.TestCase):

    def setUp(self):
        print('\n\t=> FitdistTestCase (hystat)')
        FTEST = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FTEST

    def test_fit_normal(self):

        nval = 5000
        x = np.random.normal(size=nval, loc=2, scale=0.2)

        ft = fitdist.FitDist('Power')
        ft.fit(x, nexplore=500)

        self.assertTrue(np.allclose([ft.mu], [2], atol=1e0))
        self.assertTrue(np.allclose([ft.sigma], [0.2], atol=1e-1))

 
    def test_fit_lognormal(self):

        nval = 5000
        x = np.exp(np.random.normal(size=nval, loc=2, scale=0.2))

        ft = fitdist.FitDist('Power')
        ft.fit(x, nexplore=500)

        self.assertTrue(np.allclose([ft.mu], [2], atol=1e0))
        self.assertTrue(np.allclose([ft.sigma], [0.2], atol=1e-1))


if __name__ == "__main__":
    unittest.main()


