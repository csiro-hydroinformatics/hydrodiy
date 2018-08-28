import os, re, math, sys
import unittest
from itertools import product as prod
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvt

from hydrodiy.data.qualitycontrol import ismisscens

from hydrodiy.stat.censored import normcensfit1d, normcensloglike, \
                                normcensloglike2d, \
                                normcensfit2d


class CensoredTestCase(unittest.TestCase):

    def setUp(self):
        source_file = os.path.abspath(__file__)
        ftest = os.path.dirname(source_file)
        self.ftest = ftest


    def test_normcensloglike2d(self):
        ''' Test 2d censored log-likelihood '''

        mu = np.array([0.5, 0.8])
        rho = 0.8
        scales = np.array([3, 5])
        Sig = np.array([[scales[0]**2, np.prod(scales)*rho], \
                        [np.prod(scales)*rho, scales[1]**2]])

        data = [[0, 0], [1, 0], [0, 1], [1, 1]]
        censor = 0

        nsamples = 10000000
        smp = np.random.multivariate_normal(size=nsamples, \
                            mean=mu, cov=Sig)

        # Sample probabilities
        icens = ismisscens(smp, censor)
        P0 = float(np.sum(icens==4))/nsamples

        # Compare with code
        dx = 1e-2
        for i, d in enumerate(data):
            Y = np.array(d)[None, :]
            ic = ismisscens(Y)[0]
            ll = normcensloglike2d(Y, mu, Sig, censor)
            P = math.exp(ll)

            if ic == 4:
                # both censored
                Pe = P0

            elif ic in [5, 7]:
                # one censored, not the other
                cens = 2-i
                nocens = i-1
                idx = (np.abs(smp[:, nocens]-Y[0, nocens]) < dx/2) & \
                                (smp[:, cens] < censor)
                Pe = float(np.sum(idx))/nsamples/dx

            else:
                # no censoring (wider selection bounds)
                idx = np.max(np.abs(smp-Y), 1) < 5*dx/2
                Pe = float(np.sum(idx))/nsamples/(5*dx)**2

            self.assertTrue(np.isclose(P, Pe, rtol=0., atol=5e-3))


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



    def test_normcensfit2d(self):
        ''' Test the fitting of normal censored data '''
        # Generate data
        mu = np.array([-0.5, 0.])
        scales = np.array([0.5, 2])
        rho = 0.9
        Sig = np.diag(scales**2)
        cv = np.prod(scales)*rho
        Sig[0, 1] = cv
        Sig[1, 0] = cv
        nsamples = 100

        # Run estimation
        print('\n')
        for censor in [-0.5, 0., 0.5]:
            rhoe = []
            for isample in range(nsamples):
                if isample % 20 == 0:
                    print('normcensfit2d - censor={0} isample={1}'.format(\
                                censor, isample))

                X = np.random.multivariate_normal(size=10000, \
                            mean=mu, cov=Sig)
                _, _, r = normcensfit2d(X, censor=censor)
                rhoe.append(r)

            rhoe = np.array(rhoe)
            rhom = np.mean(rhoe)
            # There appears to be a bias in the estimation...
            passing = np.isclose(rho, rhom, rtol=0., atol=5e-2)
            print(('normcensfit2d - censor={0} rho={1:0.4f} rhom={2:0.4f}'+\
                    ' => passing={3}').format(censor, rho, rhom, passing))
            self.assertTrue(passing)


if __name__ == "__main__":
    unittest.main()
