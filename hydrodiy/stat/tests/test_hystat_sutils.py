import os, math

import unittest
import numpy as np

from scipy.special import kolmogorov
from scipy.linalg import toeplitz
from scipy.stats import norm, anderson

import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import sutils
from hydrodiy.stat.sutils import HAS_C_STAT_MODULE

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


    def test_acf_error(self):
        data = np.random.uniform(size=20)
        try:
            acf, cov = sutils.acf(data, idx=(data>0.5)[2:])
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected idx'))
        else:
            raise Exception('Problem with error handling')


    def test_acf_all(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 100000
        rho = 0.8
        sig = 2
        innov = np.random.normal(size=nval, scale=sig*math.sqrt(1-rho**2))
        x = sutils.armodel_sim(rho, innov)

        maxlag = 10
        acf, cov = sutils.acf(x, maxlag)

        # Theoretical ACF for AR1 process
        expected = rho**np.arange(1, maxlag+1)
        self.assertTrue(np.allclose(acf, expected, atol=2e-2))


    def test_acf_r(self):
        for i in range(1, 5):
            fd = os.path.join(self.ftest, 'data', 'acf{0}_data.csv'.format(i))
            data, _ = csv.read_csv(fd)
            data = np.squeeze(data.values)

            fr = os.path.join(self.ftest, 'data', 'acf{0}_result.csv'.format(i))
            expected, _ = csv.read_csv(fr)
            expected = expected['acf'].values[1:]

            acf, cov = sutils.acf(data, expected.shape[0])

            # Modify for bias
            nval = len(data)
            nacf = len(acf)
            factor = np.arange(nval-nacf, nval)[::-1].astype(float)/nval
            acf *= factor

            ck = np.allclose(expected, acf)
            self.assertTrue(ck)


    def test_acf_idx(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 5000
        sig = 2
        rho1 = 0.7
        innov = np.random.normal(size=nval//2, scale=sig*math.sqrt(1-rho1**2))
        x1 = 10*sig + sutils.armodel_sim(rho1, innov)

        rho2 = 0.1
        innov = np.random.normal(size=nval//2, scale=sig*math.sqrt(1-rho2**2))
        x2 = -10*sig + sutils.armodel_sim(rho2, innov)

        data = np.concatenate([x1, x2])

        acf1, cov = sutils.acf(data, idx=data>=0)
        acf2, cov = sutils.acf(data, idx=data<=0)
        self.assertTrue(np.allclose([acf1[0], acf2[0]], [rho1, rho2], \
                            atol=1e-2))


    def test_armodel_sim(self):
        ''' Tesdt armodel_sim '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 10
        params = 0.9
        simini = 10

        # Check 1d and 2d config
        for ncol in [1, 4]:
            if ncol == 1:
                innov = np.random.normal(size=nval)
            else:
                innov = np.random.normal(size=(nval, ncol))

            outputs = sutils.armodel_sim(params, innov, simini)
            for j in range(ncol):
                y0 = simini
                expected = np.zeros(nval)
                for i in range(nval):
                    if ncol == 1:
                        expected[i] = params*y0 + innov[i]
                    else:
                        expected[i] = params*y0 + innov[i, j]

                    y0 = expected[i]

                if ncol == 1:
                    self.assertTrue(np.allclose(outputs, expected))
                else:
                    self.assertTrue(np.allclose(outputs[:, j], expected))


    def test_armodel_residual(self):
        ''' Test armodel_residual '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 10
        params = 0.9
        simini = 10

        # Check 1d and 2d config
        for ncol in [1, 4]:
            if ncol == 1:
                innov = np.random.normal(size=nval)
            else:
                innov = np.random.normal(size=(nval, ncol))

            outputs = sutils.armodel_sim(params, innov, simini)
            innov2 = sutils.armodel_residual(params, outputs, simini)

            for j in range(ncol):
                prev = simini
                expected = np.zeros(nval)
                for i in range(nval):
                    if ncol == 1:
                        value = outputs[i]
                    else:
                        value = outputs[i, j]

                    expected[i] = value-params*prev
                    prev = value

                if ncol == 1:
                    self.assertTrue(np.allclose(innov2, expected))
                else:
                    self.assertTrue(np.allclose(innov2[:, j], expected))


    def test_armodel_forward_backward(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 100
        params = 0.9
        yini = 10

        # Check 1d and 2d config
        for ncol in [10]: #[1, 10]:
            if ncol == 1:
                innov0 = np.random.normal(size=nval)
            else:
                innov0 = np.random.normal(size=(nval, ncol))

            y = sutils.armodel_sim(params, innov0, yini)
            self.assertEqual(innov0.shape, y.shape)

            innov = sutils.armodel_residual(params, y, yini)
            y2 = sutils.armodel_sim(params, innov, yini)
            self.assertTrue(np.allclose(y, y2))


    def test_armodel_nan(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 20
        innov = np.random.normal(size=nval)
        innov[5:10] = np.nan
        params = 0.9
        yini = 10

        # AR1 keeps last value before nans
        y = sutils.armodel_sim(params, innov, yini)
        y2 = sutils.armodel_sim(params, innov[10:], y[4]*params**5)
        self.assertTrue(np.allclose(y[10:], y2))

        innov = sutils.armodel_residual(params, y, yini)
        innov2 = sutils.armodel_residual(params, y[10:], innov[4]*params**5)
        self.assertTrue(np.allclose(innov[11:], innov2[1:]))


    def test_lhs_error(self):
        ''' Test latin-hypercube sampling errors '''
        nparams = 10
        nsamples = 50
        pmin = np.ones(nparams)
        pmax = 10.*np.ones(nparams)

        try:
            samples = sutils.lhs(nsamples, pmin, pmax[:2])
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected pmax'))
        else:
            raise Exception('Problem with error handling')


    def test_lhs(self):
        ''' Test latin-hypercube sampling '''
        nparams = 10
        nsamples = 50
        pmin = np.ones(nparams)
        pmax = 10.*np.ones(nparams)

        samples = sutils.lhs(nsamples, pmin, pmax)

        for i in range(nparams):
            u = (np.sort(samples[:,i])-pmin[i])/(pmax[i]-pmin[i])
            ff = sutils.ppos(nsamples)

            # Perform two sided KS test on results
            D = np.max(np.abs(u-ff))
            p = kolmogorov(D*math.sqrt(nsamples))

            self.assertTrue(p>0.95)


    def test_lhs_norm(self):
        ''' Test lhs for mvt data '''
        nsamples = 50000
        nvars = 5
        mean = np.linspace(1, 2, nvars)
        rho = 0.95
        cov = toeplitz(rho**np.arange(nvars))

        samples = sutils.lhs_norm(nsamples, mean, cov)
        meanS = np.mean(samples, axis=0)
        covS = np.cov(samples.T)

        self.assertTrue(np.allclose(mean, meanS, atol=0, \
                            rtol=1e-4))

        self.assertTrue(np.allclose(cov, covS, atol=0, \
                            rtol=1e-2))


    def test_standard_normal(self):
        ''' Test semicorr '''
        nsamples = 10
        samples = np.random.uniform(size=nsamples)
        ranks, unorm = sutils.standard_normal(samples)

        kk = np.argsort(np.argsort(samples))
        expected = norm.ppf((kk+1)/(nsamples+1))
        self.assertTrue(np.allclose(unorm, expected))

        samples = np.sort(samples)
        _, unorm1 = sutils.standard_normal(samples)
        _, unorm2 = sutils.standard_normal(samples, sorted=True)
        self.assertTrue(np.allclose(unorm1, unorm2))


    def test_semicorr(self):
        ''' Test semicorr '''
        nsamples = 50000
        nvars = 2
        rho_true = 0.55
        mean = np.array([0]*nvars)
        cov = np.array([[1, rho_true], [rho_true, 1]])
        samples = sutils.lhs_norm(nsamples, mean, cov)
        r1, u1 = sutils.standard_normal(samples[:, 0])
        r2, u2 = sutils.standard_normal(samples[:, 1])
        unorm = np.column_stack([u1, u2])

        rho, eta, rho_p, rho_m = sutils.semicorr(unorm)

        self.assertTrue(np.isclose(eta, 0.311, atol=1e-2))
        self.assertTrue(np.isclose(rho_p, 0.311, atol=1e-2))
        self.assertTrue(np.isclose(rho_m, 0.311, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
