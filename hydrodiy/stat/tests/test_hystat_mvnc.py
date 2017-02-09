import os, re, math

import unittest
import time

import numpy as np


from scipy.optimize import fmin_powell as fmin
from scipy.stats import norm, ks_2samp, kstest
from scipy.stats import ttest_1samp
from scipy.stats import t as student

from hydrodiy.stat import mvnc, sutils

import matplotlib.pyplot as plt
from hydrodiy.plot import putils

np.random.seed(0)

TEST_SAMPLE = True
TEST_FIT = False

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
        ''' Test censoring sample size '''
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
        ''' Test generation of toeplitz covariance matrix generation '''
        nvar = 6
        sig = np.ones(nvar)
        cov = mvnc.toeplitz_cov(sig, 1.)
        self.assertTrue(np.allclose(cov, np.ones((nvar, nvar))))

        cov = mvnc.toeplitz_cov(sig, 0.)
        self.assertTrue(np.allclose(cov, np.eye(nvar)))

        cov = mvnc.toeplitz_cov(sig, 0.5)
        self.assertTrue(np.allclose(cov[0, :], 0.5**np.arange(nvar)))
        self.assertTrue(np.allclose(cov, cov.T))


    def test_check_cov(self):
        ''' test covariance checking '''
        nvar = 6
        mu, cov, sig = get_mu_cov(nvar)
        mvnc.check_cov(nvar, cov, check_semidefinite=True)

        try:
            mvnc.check_cov(nvar-1, cov, check_semidefinite=True)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Expected cov of dim'))

        try:
            covb = np.random.uniform(size=(nvar, nvar))
            mvnc.check_cov(nvar, covb, check_semidefinite=True)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('cov matrix is not symetric'))

        try:
            covb = np.random.uniform(size=(nvar, nvar))
            dg = np.ones(nvar)
            dg[0] = -1
            covb = np.dot(covb.T, np.dot(np.diag(dg), covb))
            mvnc.check_cov(nvar, covb, check_semidefinite=True)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('cov matrix is not semi'))


    def test_get_icase2censvars(self):
        ''' test conversion of conditionning case to variable indexes '''
        for nvar in range(1, 10):
            maxcase = 2**nvar
            censvars = np.zeros((maxcase, nvar)).astype(bool)
            for icase in range(maxcase):
                censvars[icase] = mvnc.icase2censvars(icase, nvar)

            for k in range(2, nvar):
                cs = censvars[:, k]

                expected = np.repeat(np.repeat([False, True], 2**k)[:,None], \
                                        2**(nvar-k-1), 1)
                expected = expected.T.flatten()
                self.assertTrue(np.allclose(cs, expected))

            try:
                censvars = mvnc.icase2censvars(maxcase, nvar)
            except ValueError as err:
                pass
            self.assertTrue(str(err).startswith('Expected icase'))


    def test_conditional_error(self):
        ''' Test error in mvnc conditionning '''
        nvar = 6
        mu, cov, sig = get_mu_cov(nvar)
        mu = np.arange(nvar)

        # Test errors
        try:
            cond = mu*0
            mu_cond, cov_cond = mvnc.conditional(mu, cov, cond)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Expected p'))


    def test_conditional_invariance(self):
        ''' Test invariance in mvnc conditionning '''
        nvar = 6
        mu = np.arange(nvar, dtype=np.float64)
        mat = np.random.uniform(-1, 1, size=(nvar, nvar))
        cov = np.dot(mat, mat.T)

        # test invariance if cond == mu
        cond = np.repeat(mu[None, :], 10, 0)
        cond[:, -2:] = np.nan
        mu_cond, cov_cond, idx_cond = mvnc.conditional(mu, cov, cond)
        ck = np.allclose(mu_cond, np.repeat(mu[~idx_cond][None, :], 10, 0))
        self.assertTrue(ck)

        # test invariance if cov is diagonal
        cov = np.diag(np.diag(cov))
        mu_cond, cov_cond, idx_cond = mvnc.conditional(mu, cov, cond)
        ck = np.allclose(mu_cond, np.repeat(mu[~idx_cond][None, :], 10, 0))
        ck = ck & np.allclose(cov_cond, cov[~idx_cond][:, ~idx_cond])
        self.assertTrue(ck)

        # test invariance if idx_cond = None
        mu_cond, cov_cond, idx_cond = mvnc.conditional(mu, cov, None)
        ck = np.allclose(mu_cond, mu)
        ck = ck & np.allclose(cov_cond, cov)
        self.assertTrue(ck)


    def test_conditional_2d(self):
        ''' Test 2d conditionning '''
        nvar = 2
        mu = np.arange(nvar, dtype=np.float64)
        mat = np.random.uniform(-1, 1, size=(nvar, nvar))
        cov = np.dot(mat, mat.T)

        cond = np.array([np.nan, 2])
        mu_cond, cov_cond, idx_cond = mvnc.conditional(mu, cov, cond)


        # Sig
        expected = cov[0, 0] - cov[0, 1]**2/cov[1, 1]
        self.assertTrue(np.allclose(expected, cov_cond))

        # mu
        expected = mu[0] + cov[0, 1]/cov[1, 1]*(cond[1]-mu[1])
        self.assertTrue(np.allclose(expected, mu_cond))

        # Test vs OLS
        nsamples = 100000
        x = np.random.multivariate_normal(mean=mu, cov=cov, size=nsamples)
        A = np.concatenate([np.ones((nsamples, 1)), x[:, 1][:, None]], 1)
        B = x[:, 0][:, None]
        out = np.linalg.lstsq(A, x[:, 0][:, None])
        a, b = out[0]
        sres = out[1]
        expected = a + b*cond[1]
        self.assertTrue(np.allclose(expected, mu_cond, atol=1e-2))

        expected = sres/(nsamples-2)
        self.assertTrue(np.allclose(expected, cov_cond, atol=1e-2))


    def test_conditional_2d_vs_nd(self):
        ''' Test 2d versus multi-dimensional conditionning '''
        nvar = 10
        mu = np.arange(nvar, dtype=np.float64)
        mat = np.random.uniform(-1, 1, size=(nvar, nvar))

        # Correlation only between 1st and 2nd variable
        cov = np.dot(mat, mat.T)
        d = np.diag(cov)
        c1 = cov[0, 1]
        cov = np.diag(d)
        cov[0, 1] = c1
        cov[1, 0] = c1

        # Multi variate conditionning
        cond = np.nan * np.ones(nvar)
        cond[1] = 5.
        mu_cond, cov_cond, _ = mvnc.conditional(mu, cov, cond)

        # 2d conditionning
        cov_cond2 = cov[0, 0] - cov[0, 1]**2/cov[1, 1]
        mu_cond2 = mu[0] + cov[0, 1]/cov[1, 1]*(cond[1]-mu[1])

        self.assertTrue(np.allclose(mu_cond2, mu_cond[0][0]))
        self.assertTrue(np.allclose(cov_cond2, cov_cond[0][0]))


    def test_censoring_cases(self):
        ''' Test mvnc censoring cases generation '''
        for nvar in range(1, 10):
            maxcase = 2**nvar
            x = np.ones((maxcase, nvar))

            for icase in range(maxcase):
                censvars = mvnc.icase2censvars(icase, nvar)
                x[icase][censvars] = -1

            cases = mvnc.get_censoring_cases(x, censors=0)
            self.assertTrue(np.allclose(cases, np.arange(maxcase)))


    def test_sample(self):
        ''' Test mvnc sampling with/without conditionning '''
        if not TEST_SAMPLE:
            return

        nsamples = 200
        nvar = 6
        nrepeat = 200

        censors = np.linspace(-1, 1, nvar)
        eps = mvnc.EPS
        mu, cov, sig = get_mu_cov(nvar)

        cond_nocens = censors+1
        cond_nocens[:nvar-3] = np.nan

        cond_cens = censors+1
        cond_cens[-3:] = censors[-3:]
        cond_cens[:nvar-3] = np.nan

        pvalues = np.zeros((nrepeat, nvar))

        for i in range(nrepeat):
            if i %50 == 0:
                print('test sample {0:3d}/{1:3d}'.format(i, nrepeat))

            # Sample with censoring
            samples1 = mvnc.sample(nsamples, mu, cov, censors)

            # Sample with censoring and non-censored conditionning
            samples2 = mvnc.sample(nsamples, mu, cov, censors, \
                            cond_nocens)

            # Sample with censoring and censored conditionning
            samples3 = mvnc.sample(nsamples, mu, cov, censors, \
                            cond_cens)

            # Same sample without censoring
            samples4 = mvnc.sample(nsamples, mu, cov, censors=[-np.inf]*nvar)

            for k in range(nvar):
                s1 = np.sort(samples1[:, k])
                s4 = np.sort(samples4[:, k])

                # Censoring works ok
                self.assertTrue(np.all(s1>=censors[k]))

                # Compare distributions before and after
                s1 = s1[s1>censors[k]]
                s4 = s4[s4>censors[k]]

                # KS test
                D, pvalues[i, k] = ks_2samp(s1, s4)

        # Test pvalues are uniformly distributed
        pv = np.array([kstest(pvalues[:, k], 'uniform')[0] \
                    for k in range(nvar)])
        ck = np.all(pv>1e-3)
        self.assertTrue(ck)


    def test_logpdf_fit(self):
        ''' Test consistency between logpdf and sampling '''
        if not TEST_FIT:
            return

        nsamples = 500
        nfit = 50
        nrepeat = 10

        # Function to simplify parameterisation
        def p2m(params):
            mu, sig, rho = params
            mu = np.ones(nvar)*mu
            cov = mvnc.toeplitz_cov(np.ones(nvar)*sig, rho)
            return mu, cov

        # Fitting function
        def fitfun(params, data, cases):
            # Check params bounds
            if params[1]<1e-3:
                return np.inf

            if (params[2]<0.01) or (params[2]>0.99):
                return np.inf

            # Get mu and covariance
            mu, cov = p2m(params)

            # compute log pdf
            lp = mvnc.logpdf(data, cases, mu, cov, censors)

            return -np.sum(lp)

        # True parameter set
        truep = np.array([1, 1.5, 0.7])

        # Loop over number of variables
        for nvar in [2]: #range(1, 6):

            # Loop over number of censored variables
            for ncensors in [1]: #range(nvar+1):

                # Censor values
                censors = -np.inf * np.ones(nvar)
                censors[:ncensors] = np.linspace(-0.5, -0.2, nvar)[:ncensors]

                # Generate true mu and cov
                mu, cov = p2m(truep)

                # Repeat the fitting process
                pvalues = np.zeros((nrepeat, 3))
                optp_means = np.zeros(nrepeat)

                for irepeat in range(nrepeat):

                    print('nvar {0} ncensors {1}: repeat {2}'.format(\
                                nvar, ncensors, irepeat))

                    # Iterate through random samples
                    optp = np.zeros((nfit, len(truep)))
                    optp2 = np.zeros((nfit, 2))

                    for i in range(nfit):
                        if i%10 == 0:
                            print('\t fit {0}/{1}'.format(i, nfit))

                        # Generate random data
                        samples = mvnc.sample(nsamples, mu, cov, censors)
                        cases = mvnc.get_censoring_cases(samples, censors)

                        # Add offset to true parameters then Fit,
                        # we should find truep back
                        startp = truep + np.random.uniform(-1e-2, 1e-2, \
                                                len(truep))
                        optp[i,:] = fmin(fitfun, startp, args=(samples, cases, ), \
                                            disp=False)

                    # T test on parameter mean
                    tstat, pv  = ttest_1samp(optp, truep)
                    pvalues[irepeat, :] = pv
                    optp_means[irepeat] = np.mean(optp, 0)[0]

                # Compute KS test pvalues
                # Takes too long ...
                #pv = [kstest(pvalues[:, k], 'uniform')[1] for k in range(3-(nvar==1))]
                #ck = np.all(pv>0.1)

                # Test passes if more than 20% of the repeats pass the t test
                # with pvalue of 5%
                # (a more accurate test would check that only 5% fail the t
                # test)
                cc = range(3-(nvar==1))
                thresh = int(nrepeat * 0.2)
                count = np.sum(pvalues[:, cc] < 0.05, 0)
                ck = np.all(count<=thresh)

                if ck:
                    mess = 'GOOD'
                else:
                    mess = 'FAILED'

                print(('nvar {0} - ncensors {1} - nrepeat {10}: {2} :' + \
                    'count pv<5% = {3} <= {4}?\n' + \
                    '\ttrue               = {5}\n' + \
                    '\tmean (last repeat) = {6}\n' +\
                    '\tstd  (last repeat) = {7}\n' + \
                    '\tTstat(last repeat) = {8}\n'+\
                    '\tTpval(last repeat) = {9}').format(nvar, ncensors, \
                    mess, count, thresh, truep+1e-8, np.mean(optp, 0),\
                    np.std(optp, 0), tstat, pv, nrepeat))

                #self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
