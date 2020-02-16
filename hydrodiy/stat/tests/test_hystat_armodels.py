import os, math

import unittest
import numpy as np

from scipy.special import kolmogorov
from scipy.linalg import toeplitz
from scipy.stats import norm, anderson

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import armodels
from hydrodiy.stat.armodels import HAS_C_STAT_MODULE

np.random.seed(0)

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hystat)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_armodel_sim(self):
        ''' Tesdt armodel_sim '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 10
        params = np.linspace(0.9, 0.2, 10)
        simini = 10

        # Check for various orders
        for order in range(5):

            # Check single and multi columns
            for ncols in [1, 4]:
                innov = np.random.normal(size=(nval, ncols))

                # Run armodel
                outputs = sutils.armodel_sim(params[:order], innov, simini)

                # Compute expected value
                for j in range(ncols):
                    y0 = np.ones(order) * simini
                    expected = np.zeros(nval)
                    for i in range(nval):
                        expected[i] = np.sum(params[:order]*y0) \
                                            + innov[i, j]

                        for k in np.arange(order)[::-1]:
                            if k > 0:
                                y0[k] = y0[k-1]
                            else:
                                y0[k] = expected[i]

                    self.assertTrue(np.allclose(outputs[:, j], expected))


    def test_armodel_residual(self):
        ''' Test armodel_residual '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 10
        params = np.linspace(0.9, 0.2, 10)
        simini = 10

        # Check 1d and 2d config
        for order in range(5):
            for ncols in [1, 4]:
                # Generate data
                innov = np.random.normal(size=(nval, ncols))
                outputs = sutils.armodel_sim(params[:order], innov, simini)

                innov2 = sutils.armodel_residual(params[:order], outputs, simini)
                self.assertTrue(np.allclose(innov, innov2))


    def test_armodel_forward_backward(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 100
        params = np.linspace(0.9, 0.2, 10)
        yini = 10

        # Check 1d and 2d config
        for order in range(5):
            for ncols in [1, 4]:
                innov0 = np.random.normal(size=(nval, ncols))

                y = sutils.armodel_sim(params[:order], innov0, yini)
                self.assertEqual(innov0.shape, y.shape)

                innov = sutils.armodel_residual(params[:order], y, yini)
                y2 = sutils.armodel_sim(params[:order], innov, yini)
                self.assertTrue(np.allclose(y, y2))


    def test_armodel_nan(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 20
        innov = np.random.normal(size=nval)
        innov[5:10] = np.nan
        params = [0.5, 0.1]
        yini = 10

        # Run AR1
        y2 = sutils.armodel_sim(params, innov, yini, False)
        print('\n\n')
        y2 = sutils.armodel_sim(params, innov, yini, True)

        import pdb; pdb.set_trace()
        expected = sutils.armodel_sim(params, innov[10:], yini, False)


        y1 = sutils.armodel_sim(params, innov, yini, True)
        import pdb; pdb.set_trace()

        # Run AR1 with 0 innov
        innov2 = innov.copy()
        innov2[np.isnan(innov2)] = 0
        y2 = sutils.armodel_sim(params, innov2, yini)
        yini = y2[9]
        y3 = sutils.armodel_sim(params, innov2[10:], yini)

        self.assertTrue(np.allclose(y[10:], y3))

        innov = sutils.armodel_residual(params, y, yini)
        innov2 = sutils.armodel_residual(params, y[10:], innov[4]*params**5)
        self.assertTrue(np.allclose(innov[11:], innov2[1:]))



if __name__ == "__main__":
    unittest.main()
