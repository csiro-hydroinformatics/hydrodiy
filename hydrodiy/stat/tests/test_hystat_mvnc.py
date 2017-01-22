import os, re, math

import unittest
import time

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import fmin_powell as fmin
from scipy.stats import norm, ks_2samp
from scipy.stats import t as student

from hydrodiy.stat import mvnc, sutils
from hydrodiy.plot import putils

np.random.seed(0)

def get_mu_cov(nvar, rho=0.8):
    mu = 2*np.ones(nvar)
    sig = np.ones(nvar)
    cov = mvnc.toeplitz_cov(sig, rho)

    return mu, cov, sig

class MVNCTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MvtTestCase')
        testfile = os.path.abspath(__file__)
        self.ftest = os.path.dirname(testfile)


    def test_censors_size(self):
        nsamples = 10000
        nvar = 6

        censors = range(1, nvar+2)
        eps = mvnc.EPS
        mu, cov, sig = get_mu_cov(nvar)

        try:
            samples = mvnc.sample(nsamples, mu, cov, censors)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Expected'))

        try:
            samples = mvnc.sample(nsamples, mu, cov, censors[:-1])
            lp = mvnc.logpdf(samples, mu, cov, censors)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Expected'))


    def test_toeplitz(self):
        nvar = 6
        sig = np.ones(nvar)
        cov = mvnc.toeplitz_cov(sig, 1.)
        self.assertTrue(np.allclose(cov, np.ones((nvar, nvar))))

        cov = mvnc.toeplitz_cov(sig, 0.)
        self.assertTrue(np.allclose(cov, np.eye(nvar)))

        cov = mvnc.toeplitz_cov(sig, 0.5)
        self.assertTrue(np.allclose(cov[0, :], 0.5**np.arange(nvar)))
        self.assertTrue(np.allclose(cov, cov.T))


    def test_sample(self):
        nsamples = 50
        nvar = 6

        censors = np.linspace(-1, 1, nvar)
        eps = mvnc.EPS
        mu, cov, sig = get_mu_cov(nvar)

        samples1 = mvnc.sample(nsamples, mu, cov, censors)

        # Same sample without censoring
        samples2 = mvnc.sample(nsamples, mu, cov, [-np.inf]*nvar)

        for k in range(nvar):
            s1 = np.sort(samples1[:, k])
            s2 = np.sort(samples2[:, k])

            # Censoring works ok
            self.assertTrue(np.all(s1>=censors[k]))

            # Compare distributions before and after
            s1 = s1[s1>censors[k]]
            s2 = s2[s2>censors[k]]

            # KS test
            pvalue, D = ks_2samp(s1, s2)
            ck = pvalue>0.05
            self.assertTrue(ck)


    def test_logpdf_fit(self):
        nvar = 2
        nsamples = 1000
        nrepeat = 100

        for opt_censoring in
        censors = np.linspace(-1, 1, nvar)
        censors = [-np.inf] * nvar

        # Function to simplify parameterisation
        def p2m(params):
            mu, sig, rho = params
            mu = np.ones(nvar)*mu
            cov = mvnc.toeplitz_cov(np.ones(nvar)*sig, rho)
            return mu, cov

        def fit(params, data):
            # Check params bounds
            if (params[2]<0.01) or (params[2]>0.99):
                return np.inf

            # Get mu and covariance
            mu, cov = p2m(params)

            # compute log pdf
            lp = mvnc.logpdf(data, mu, cov, censors)

            return -np.sum(lp)

        # True parameter set
        params_true = np.array([2, 1, 0.9])
        mu, cov = p2m(params_true)

        # Iterate through random samples
        params_opt = np.zeros((nrepeat, len(params_true)))
        for i in range(nrepeat):
            if i%10 == 0:
                print('testing fit, iteration {0}/{1}'.format(\
                            i, nrepeat))

            # Generate random data
            samples = mvnc.sample(nsamples, mu, cov, censors)

            # Fit, we should find params_true
            params_start = [3, 2, 0.5]
            params_opt[i,:] = fmin(fit, params_start, args=(samples, ), \
                                disp=False)

        # True parameters should be close to mean
        # within uncertainty (assuming normal distribution, hence t stats)
        mup = np.mean(params_opt, axis=0)
        stdp = np.std(params_opt, axis=0)
        ci1 = params_true + stdp*student.ppf(0.25, nrepeat-1)
        ci2 = params_true + stdp*student.ppf(0.75, nrepeat-1)

        ck = np.all((mup>ci1) & (mup<ci2))
        if not ck:
            plt.close('all')
            fig, axs = plt.subplots(ncols=3)
            for k in range(3):
                axs[k].hist(params_opt[:, k])
            plt.show()
            print('No good!\n\ttrue = {0}\n\tci1 = {1}\n\tci2 = {2}\n\tmean = {3}'.format( \
                params_true, ci1, ci2, mup))

        self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
