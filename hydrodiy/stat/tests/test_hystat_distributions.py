import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.stat.distributions import lognormscensored0
from hydrodiy.stat.distributions import powernormcensored0
from hydrodiy.stat.distributions import power, powerdiff, powerinv
from hydrodiy.stat import sutils

from scipy.stats import norm
from scipy.special import erf

class LogNormCensoredTestCase(unittest.TestCase):

    def setUp(self):
        print('\n\t=> LogNormShiftedCensoredTestCase (hystat)')
        FTEST = os.path.dirname(os.path.abspath(__file__))
        self.FTEST = FTEST

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
        nval = 10000
        mu = -1.
        sig = 0.5
        shift = 0.3
        q = sutils.empfreq(nval)
        x = lognormscensored0.ppf(q, mu, sig, shift)

        # perform fit
        params = lognormscensored0.fit(x)

        ck = np.allclose(params[:3], (mu, sig, shift), rtol=5e-2)
        self.assertTrue(ck)

        # Check pvalue of diagnostics
        fd = lognormscensored0.fit_diagnostic
        self.assertTrue(fd['ks_pvalue']>0.9)
        self.assertTrue(fd['mw_pvalue']>0.9)


    def test_fit_data(self):
        fd = os.path.join(self.FTEST, 'lognormdata.csv')
        data = pd.read_csv(fd)
        obs = data.iloc[:, 1]
        params = lognormscensored0.fit(obs)
        paramstxt = ' '.join(['{0:3.3e}'.format(pp) for pp in params[:3]])

        q = sutils.empfreq(len(obs))
        sim = lognormscensored0.ppf(q, *params[:3])

        mu = params[0]
        sig = params[1]*1.1
        shift = params[2]
        simb = lognormscensored0.ppf(q, mu, sig, shift)

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(obs, q, label='obs')
        ax.plot(sim, q, label='sim {0}'.format(paramstxt))
        ax.plot(simb, q, label='simb')
        #ax.set_xscale('log')
        ax.legend(loc=4, fontsize='x-small')
        fig.savefig(os.path.join(self.FTEST, 'fit.png'))

        #ck = np.allclose(params[:3], (mu, sig, shift), rtol=0.1)
        #self.assertTrue(ck)


class PowerNormCensoredTestCase(unittest.TestCase):

    def setUp(self):
        print('\n\t=> PowerNormCensoredTestCase (hystat)')
        FTEST = os.path.dirname(os.path.abspath(__file__))
        self.FTEST = FTEST

    def test_power(self):
        x = np.linspace(-100, 100, 3000)
        for lam in np.linspace(0., 3., 100):
            for cst in np.linspace(1e-4, 2, 5):
                y = power(x, lam, cst)
                xx = powerinv(y, lam, cst)
                ck = np.allclose(xx, x)
                self.assertTrue(ck)

    def test_powerdiff(self):
        x = np.linspace(-100, 100, 3000)
        i1 = np.where(x<0)[0][-1]
        i2 = np.where(x>0)[0][0]
        dx = 1e-6
        for lam in np.linspace(1e-4, 3., 100):
            for cst in np.linspace(1e-4, 2, 5):
                y1 = power(x, lam, cst)
                y2 = power(x+dx, lam, cst)
                dy = powerdiff(x, lam, cst)
                ddy = (y2-y1)/dx
                kk = np.arange(len(x))
                idx = (kk < i1) | (kk > i2)
                # Avoids derivative around 0
                ck = np.allclose(dy[idx], ddy[idx], atol=1e-5)
                self.assertTrue(ck)

