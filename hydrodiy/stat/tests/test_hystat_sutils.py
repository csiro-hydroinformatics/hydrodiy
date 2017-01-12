import os, math

import unittest
import numpy as np

from scipy.special import kolmogorov

from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import sutils

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


    def test_ar1(self):
        nval = 100

        # Check 1d and 2d config
        for ncol in [0, 10]:
            if ncol == 0:
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
