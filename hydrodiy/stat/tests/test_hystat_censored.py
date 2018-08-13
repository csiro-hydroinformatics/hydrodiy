import os, re, math, sys
import unittest
import logging
from itertools import product as prod
import numpy as np
import pandas as pd
from scipy.stats import norm

from hydrodiy.stat.censored import normcensfit1d, normcensloglike, \
                                censindexes, normcensloglike2d


class CensoredTestCase(unittest.TestCase):

    def setUp(self):
        source_file = os.path.abspath(__file__)
        ftest = os.path.dirname(source_file)
        self.ftest = ftest

    def test_censindexes(self):
        ''' Test the fitting of normal censored data '''
        Y = np.zeros((4, 2))
        Y[1, 0] = 1
        Y[2, 1] = 1
        Y[3, :] = 1

        icens = censindexes(Y)
        self.assertTrue(np.allclose(icens, [3, 2, 1, 0]))

    def test_normcensloglike2d(self):
        ''' Test 2d censored log-likelihood '''

        mu = np.array([0.5, 0.8])
        s1, s2, rho = 2, 3, 0.8
        Sig = np.array([[s1**2, s1*s2*rho], [s1*s2*rho, s2**2]])
        data = [[0, 0], [1, 0], [0, 1], [1, 1]]
        censor = 1e-10

        nsamples = 100000
        smp = np.random.multivariate_normal(size=nsamples, \
            mean=mu, cov=Sig)

        ll = np.zeros(len(data))
        for i, d in enumerate(data):
            Y = np.array(d)[None, :]
            ll[i] = normcensloglike2d(Y, mu, Sig, censor)

        # Test all censored case
        idx = np.sum(smp < censor, axis=1)
        P = np.sum(idx)/len(idx)



    def test_normcensfit1d(self):
        ''' Test the fitting of normal censored data '''
        # Generate data
        mu = 1
        sig = 2
        x = np.random.normal(size=10000, loc=mu, scale=sig)
        x = np.sort(x)

        # Run estimation
        params = []
        for censor in [-100] + list(np.linspace(mu-sig/2, mu+sig/2, 10)):
            params.append(normcensfit1d(x, censor=censor, sort=False))

        params = np.row_stack(params)
        self.assertTrue(np.allclose(params[:, 0], mu, rtol=0., atol=1e-1))
        self.assertTrue(np.allclose(params[:, 1], sig, rtol=0., atol=1e-1))


    def test_normcensfit1d_error(self):
        ''' Test normcensfit1d errors '''
        x = np.random.uniform(0, 1, size=100)
        x[0] = np.nan
        try:
            mu, sig = normcensfit1d(x, censor=0., sort=False)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected no nan'))
        else:
            raise ValueError('Problem with error handling')


    def test_normcensfit1d_zeros(self):
        ''' Test the fitting of normal censored data with high number of
            censored values
        '''
        # Generate data
        mu = 1
        sig = 2
        nval = 30
        ff = (np.arange(1, nval+1)-0.3)/(nval+0.4)
        x = mu+sig*norm.ppf(ff)
        x = np.sort(x)

        censor = x[-2]
        mu, sig = normcensfit1d(x, censor=censor, sort=False)
        self.assertTrue(np.isclose(mu, -0.86973, rtol=0., atol=1e-4))
        self.assertTrue(np.isclose(sig, 2.75349, rtol=0., atol=1e-4))

        censor = x[-1]
        mu, sig = normcensfit1d(x, censor=censor, sort=False)
        self.assertTrue(np.isclose(mu, -2.24521, rtol=0., atol=1e-4))
        self.assertTrue(np.isclose(sig, 3.39985, rtol=0., atol=1e-4))


    def test_normcensloglike(self):
        ''' Test censored log likelihood '''

        x = np.random.normal(size=10000)
        params = []
        nexplore = 50

        for censor in np.linspace(-1, 1, 5):
            # Explore parameter space and compute log likelihood
            ll = np.zeros((nexplore**2, 3))
            for i, (mu, sig) in enumerate(prod(\
                            np.linspace(-1, 1, nexplore), \
                            np.linspace(1e-1, 2, nexplore))):
                ll[i, :] = [mu, sig, normcensloglike(x, mu, sig, censor)]

            # max likelihood
            imaxll = np.argmax(ll[:, 2])

            # distance between parameters and true parameters
            dist = np.sqrt(np.sum(np.abs(ll[:, :2] \
                        - np.array([0, 1])[None, :])**2, 1))

            # Check maxlikelihood is within acceptable distance
            imind = np.where(dist < 1e-1)[0]
            self.assertTrue(imaxll in imind)


if __name__ == "__main__":
    unittest.main()
