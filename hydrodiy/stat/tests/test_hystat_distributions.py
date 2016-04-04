import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.stat.distributions import lognormscensored0

from scipy.stats import norm
from scipy.special import erf

class LogNormCensoredTestCase(unittest.TestCase):

    def setUp(self):
        print('\n\t=> LogNormShiftedCensoredTestCase (hystat)')
        FTEST = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FTEST

    def test_pdf(self):
        mu = 0.
        sig = 1.
        shift = 0.5
        x = np.linspace(0.1, 3., 10)

        res = lognormscensored0.pdf(x, mu, sig, shift)
        expected = np.exp(-(np.log(x+shift)-mu)**2/2/sig**2)
        expected = expected/(x+shift)/sig/math.sqrt(2*math.pi)
        self.assertTrue(np.allclose(res, expected))

        res = lognormscensored0.pdf(0., mu, sig, shift)
        self.assertTrue(np.allclose(res, np.inf))

        res = lognormscensored0.pdf(-1, mu, sig, shift)
        self.assertTrue(np.allclose(res, 0.))

    def test_cdf(self):
        mu = 0.
        sig = 1.
        shift = 0.5
        x = np.linspace(0.1, 3., 10)

        res = lognormscensored0.cdf(x, mu, sig, shift)
        expected = 0.5*(1+erf((np.log(x+shift)-mu)/math.sqrt(2)/sig))
        self.assertTrue(np.allclose(res, expected))

        res = lognormscensored0.cdf(1e-10, mu, sig, shift)
        expected = 0.5*(1+erf((np.log(shift)-mu)/math.sqrt(2)/sig))
        self.assertTrue(np.allclose(res, expected))

        res = lognormscensored0.cdf(-1, mu, sig, shift)
        self.assertTrue(np.allclose(res, 0.))


    def test_ppf(self):
        mu = 0.
        sig = 1.
        shift = 0.5
        x = np.linspace(0., 3., 10)

        res = lognormscensored0.ppf(0., mu, sig, shift)
        self.assertTrue(np.allclose(res, 0.))

        q = 1e-2
        res = lognormscensored0.ppf(q, mu, sig, shift)
        expected = 0.
        self.assertTrue(np.allclose(res, expected))

        q = 0.5
        res = lognormscensored0.ppf(q, mu, sig, shift)
        expected = np.exp(norm.ppf(q)*sig+mu)-shift
        self.assertTrue(np.allclose(res, expected))

        q = 1-1e-2
        res = lognormscensored0.ppf(q, mu, sig, shift)
        expected = np.exp(norm.ppf(q)*sig+mu)-shift
        self.assertTrue(np.allclose(res, expected))

        q = np.linspace(1e-1, 1, 1e-1)
        res = lognormscensored0.ppf(q, mu, sig, shift)
        expected = np.exp(norm.ppf(q)*sig+mu)-shift
        expected[expected < 0.] = 0.
        self.assertTrue(np.allclose(res, expected))


    def test_fit(self):
        nval = 5000
        mu = -1.
        sig = 2.
        shift = 0.1
        x = np.exp(np.random.normal(mu, sig, size=nval))-shift
        x[x<0.] = 0.

        params = lognormscensored0.fit(x)
        ck = np.allclose(params[:3], (mu, sig, shift), rtol=0.1)
        self.assertTrue(ck)

