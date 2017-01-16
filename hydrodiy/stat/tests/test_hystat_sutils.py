import os, math

import unittest
import numpy as np

from scipy.special import kolmogorov

from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import sutils

np.random.seed(0)

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hystat)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_ppos(self):
        nval = 100

        pp = sutils.ppos(nval)
        self.assertTrue(pp[0]>0)
        self.assertTrue(pp[-1]<1)
        ppd = np.diff(pp)
        self.assertTrue(np.all(ppd>0))

        pp = sutils.ppos(nval, 0.)
        self.assertTrue(np.allclose(pp, np.arange(1., nval+1.)/(nval+1)))


    def test_acf(self):
        nval = 100000
        rho = 0.8
        sig = 2
        innov = np.random.normal(size=nval, scale=sig*math.sqrt(1-rho**2))
        x = sutils.ar1innov([rho, 0.], innov)

        maxlag = 10
        acf = sutils.acf(x, maxlag)
        # Theoretical ACF for AR1 process
        expected = rho**np.arange(1, maxlag+1)
        self.assertTrue(np.allclose(acf, expected, atol=1e-2))


    def test_ar1_forward(self):
        nval = 10
        params = np.array([0.9, 10])

        # Check 1d and 2d config
        for ncol in [1, 4]:
            if ncol == 1:
                innov = np.random.normal(size=nval)
            else:
                innov = np.random.normal(size=(nval, ncol))

            outputs = sutils.ar1innov(params, innov)
            for j in range(ncol):
                alpha, y0 = params
                expected = np.zeros(nval)
                for i in range(nval):
                    if ncol == 1:
                        expected[i] = alpha*y0 + innov[i]
                    else:
                        expected[i] = alpha*y0 + innov[i, j]

                    y0 = expected[i]

                if ncol == 1:
                    self.assertTrue(np.allclose(outputs, expected))
                else:
                    self.assertTrue(np.allclose(outputs[:, j], expected))


    def test_ar1_backward(self):
        nval = 10
        params = np.array([0.9, 10])

        # Check 1d and 2d config
        for ncol in [1, 4]:
            if ncol == 1:
                innov = np.random.normal(size=nval)
            else:
                innov = np.random.normal(size=(nval, ncol))

            outputs = sutils.ar1innov(params, innov)
            innov2 = sutils.ar1inverse(params, outputs)

            for j in range(ncol):
                alpha, y0 = params

                expected = np.zeros(nval)
                for i in range(nval):
                    if ncol == 1:
                        value = outputs[i]
                    else:
                        value = outputs[i, j]

                    expected[i] = value-alpha*y0
                    y0 = value

                if ncol == 1:
                    self.assertTrue(np.allclose(innov2, expected))
                else:
                    self.assertTrue(np.allclose(innov2[:, j], expected))


    def test_ar1_forward_backward(self):
        nval = 100

        # Check 1d and 2d config
        for ncol in [1, 10]:
            if ncol == 1:
                innov0 = np.random.normal(size=nval)
            else:
                innov0 = np.random.normal(size=(nval, ncol))

            params = np.array([0.9, 10])
            y = sutils.ar1innov(params, innov0)
            self.assertEqual(innov0.shape, y.shape)

            innov = sutils.ar1inverse(params, y)
            y2 = sutils.ar1innov(params, innov)
            self.assertTrue(np.allclose(y, y2))


    def test_lhs(self):
        nparams = 10
        nsamples = 50
        pmin = 1
        pmax = 10

        samples = sutils.lhs(nparams, nsamples, pmin, pmax)

        for i in range(nparams):
            u = (np.sort(samples[:,i])-pmin)/(pmax-pmin)
            ff = sutils.ppos(nsamples)

            # Perform two sided KS test on results
            D = np.max(np.abs(u-ff))
            p = kolmogorov(D*math.sqrt(nsamples))

            self.assertTrue(p>0.95)


if __name__ == "__main__":
    unittest.main()
