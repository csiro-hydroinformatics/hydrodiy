import os, re, math

import unittest
import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

        # Sample with censoring
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
        nsamples = 1000
        nrepeat = 100

        # Function to simplify parameterisation
        def p2m(params):
            mu, sig, rho = params
            mu = np.ones(nvar)*mu
            cov = mvnc.toeplitz_cov(np.ones(nvar)*sig, rho)
            return mu, cov

        # Fitting function
        def fit(params, data):
            # Check params bounds
            if (params[2]<0.01) or (params[2]>0.99):
                return np.inf

            # Get mu and covariance
            mu, cov = p2m(params)

            # compute log pdf
            lp = mvnc.logpdf(data, mu, cov, censors)

            return -np.sum(lp)

        # Loop over number of variables
        for nvar in range(1, 5):

            # Loop over number of censored variables
            for ncensors in range(nvar+1):

                # Censor values
                censors = -np.inf * np.ones(nvar)
                censors[:ncensors] = np.linspace(0, 0.5, nvar)[:ncensors]

                # True parameter set
                truep = np.array([1, 1.5, 0.7])
                mu, cov = p2m(truep)

                # Iterate through random samples
                optp = np.zeros((nrepeat, len(truep)))
                for i in range(nrepeat):
                    if i%10 == 0:
                        print('testing fit, nvar {3} ncensors {2} - iteration {0}/{1}'.format(\
                                    i, nrepeat, ncensors, nvar))

                    # Generate random data
                    samples = mvnc.sample(nsamples, mu, cov, censors)

                    # Start slightly of the true parameters then Fit,
                    # we should find truep back
                    startp = truep + 1e-2
                    optp[i,:] = fmin(fit, startp, args=(samples, ), \
                                        disp=False)

                    #mu, cov = p2m(optp[i, :])
                    #plt.close('all')
                    #fig, axs = plt.subplots(ncols=2, nrows=2)
                    #axs[0, 0].hist(samples[:, 0])
                    #axs[0, 1].hist(samples[:, 0])
                    #axs[1, 0].plot(samples[:, 0], samples[:, 1], 'o')
                    #for pvalue in [0.5, 0.95]:
                    #    el = putils.cov_ellipse(mu, cov, facecolor='none')
                    #    axs[1, 0].add_patch(el)
                    #axs[1, 0].add_patch(el)

                    #plt.show()

                # True parameters should be close to mean
                # within uncertainty (assuming normal distribution, hence t stats)
                mup = np.mean(optp, axis=0)
                stdp = np.std(optp, axis=0)
                ci1 = mup + stdp*student.ppf(0.05, nrepeat-1)
                ci2 = mup + stdp*student.ppf(0.95, nrepeat-1)

                # Remove last parameter if nvar == 1
                ck = np.all(((truep>ci1) & (truep<ci2))[:nvar-(nvar==1)])
                mess = 'GOOD'
                if not ck:
                    mess = 'ERROR'
                    plt.close('all')
                    fig, axs = plt.subplots(ncols=3)
                    for k in range(3):
                        ax = axs[k]
                        ax.hist(optp[:, k])
                        putils.line(ax, 0, 1, truep[k], 0., 'r-', lw=2)
                    plt.show()

                print('{4}\n\ttrue = {0}\n\tci1 = {1}\n\tci2 = {2}\n\tmean = {3}\n'.format( \
                        truep, ci1, ci2, mup, mess))

                self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
