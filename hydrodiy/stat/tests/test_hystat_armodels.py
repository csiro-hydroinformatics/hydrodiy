import os, math

import unittest
import numpy as np

from scipy.special import kolmogorov
from scipy.linalg import toeplitz
from scipy.stats import norm, anderson

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import sutils
from hydrodiy.stat.armodels import HAS_C_STAT_MODULE, \
                                armodel_sim, armodel_residual, \
                                yule_walker

np.random.seed(0)

class ARModelsTestCase(unittest.TestCase):

    def setUp(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest("Missing C module c_hydrodiy_stat")

        print("\t=> UtilsTestCase (hystat)")
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_armodel_sim(self):
        """ Test armodel_sim """
        nval = 100
        params = np.linspace(0.9, 0.2, 10)
        sim_mean = 5
        sim_ini = 10

        # Check for various orders
        for order in range(1, 5):
            innov = np.random.normal(size=nval)

            # Run armodel
            outputs = armodel_sim(params[:order], innov, \
                                    sim_mean, sim_ini)

            # Compute expected value
            prev_centered = np.ones(order) * (sim_ini-sim_mean)
            expected = np.zeros(nval)
            for i in range(nval):
                expected[i] = np.sum(params[:order]*prev_centered) \
                                    + innov[i]

                for k in np.arange(order)[::-1]:
                    if k > 0:
                        prev_centered[k] = prev_centered[k-1]
                    else:
                        prev_centered[k] = expected[i]

            expected += sim_mean
            self.assertTrue(np.allclose(outputs, expected))


    def test_armodel_forward_backward(self):
        """ Test back and forth with armodels """
        nval = 100
        params = np.linspace(0.9, 0.2, 10)
        ymean = 20
        yini = 10

        # Check 1d and 2d config
        for order in range(1, 5):
            innov0 = np.random.normal(size=nval)

            y = armodel_sim(params[:order], innov0, ymean, yini)
            self.assertEqual(innov0.shape, y.shape)
            residuals = armodel_residual(params[:order], y, ymean, yini)

            y2 = armodel_sim(params[:order], residuals, ymean, yini)
            self.assertTrue(np.allclose(y, y2))


    def test_armodel_sim_nan(self):
        """ Test armodel simulation with nan """
        nval = 20
        innov = np.random.normal(size=nval)
        innov[5:10] = np.nan
        params = [0.5, 0.1]
        yini = 10

        y = armodel_sim(params, innov, yini)

        # Compare with ar model run with nan innov set to 0
        innov[np.isnan(innov)] = 0
        expected = armodel_sim(params, innov, yini)
        self.assertTrue(np.allclose(y, expected))


    def test_armodel_residual_nan(self):
        """ Test armodel simulation with nan """
        nval = 20
        innov = np.random.normal(size=nval)
        params = np.array([0.5, 0.1])
        nparams = len(params)
        yini = 10
        ymean = 20
        y = armodel_sim(params, innov, ymean, yini)
        y[5:10] = np.nan

        residuals = armodel_residual(params, y, ymean, yini)

        # compute expected
        expected = 0*y
        prev_centered = np.ones(nparams) * (yini-ymean)

        for i in range(nval):
            value = y[i]-ymean

            if np.isnan(value):
                value = np.sum(params*prev_centered)

            expected[i] = value-np.sum(params*prev_centered)

            for k in np.arange(len(params))[::-1]:
                if k > 0:
                    prev_centered[k] = prev_centered[k-1]
                else:
                    prev_centered[k] = value

        self.assertTrue(np.allclose(residuals, expected))


    def test_yule_walker(self):
        """ Test yule-walker """
        nval = 100000
        innov = np.random.normal(size=nval)
        params = np.array([0.8, -0.1])
        yini = 10
        ymean = 20
        y = armodel_sim(params, innov, ymean, yini)

        acf, _ = sutils.acf(y, maxlag=3)
        params = yule_walker(acf)

        # Not sure about this test
        # TODO check it makes sense
        self.assertTrue(np.allclose(params, \
                            [0.67, -0.02], \
                            rtol=0., atol=1e-2))


if __name__ == "__main__":
    unittest.main()
