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
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_empfreq(self):
        nval = 10
        prob_cst = 0.2
        expected = (np.linspace(1, nval, nval)-prob_cst)/(nval+1-2*prob_cst)
        freq = sutils.empfreq(nval, prob_cst)
        self.assertTrue(np.allclose(freq, expected))

    def test_percentiles(self):
        nval = 10
        x = np.linspace(0, 1, nval)
        qq = np.linspace(0,100, 5)

        xq = sutils.percentiles(x, qq, 1., 0)
        self.assertTrue(np.allclose(xq*100, qq))
        self.assertEqual(list(xq.index), [str(int(q))+'%' for q in qq])

        qq = np.linspace(0,100, 20)
        xq = sutils.percentiles(x, qq, digits=2)
        self.assertEqual(list(xq.index), ['{0:0.2f}%'.format(q) for q in qq])


    def test_acf1(self):
        fdata = '%s/data/acf1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        data = data.astype(float)
        fres = '%s/data/acf1_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))
        self.assertTrue(np.allclose(res['acf'].values,
            expected['acf'].values))

    def test_acf2(self):
        fdata = '%s/data/acf2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        fres = '%s/data/acf2_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))

        self.assertTrue(np.allclose(res['acf'].values,
            expected['acf'].values))

    def test_acf3(self):
        fdata = '%s/data/acf1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        data = data.astype(float).values
        data = np.vstack([data[:5], 1000.,  np.nan, data[5:],
                    np.nan, np.nan])

        fres = '%s/data/acf1_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        filter = np.array([True]*len(data))
        filter[5] = False

        res = sutils.acf(data, lag=range(0,6), filter=filter)
        self.assertTrue(np.prod(np.abs(res['acf'])<=1+1e-10)==1)

    def test_ar1(self):
        nval = 10
        innov0 = np.random.normal(size=nval)

        params = np.array([0.9, 10])
        y = sutils.ar1innov(params, innov0)

        innov = sutils.ar1inverse(params, y)
        y2 = sutils.ar1innov(params, innov)
        self.assertTrue(np.allclose(y, y2))

    def test_pit(self):

        nval = 100
        nens = 1000

        pit1 = np.round(np.random.uniform(0, 1, nval), 3)

        ff = sutils.empfreq(nens)
        forc = np.random.uniform(0, 100, (nval, nens))
        obs = np.ones((nval,)) * np.nan

        for i in range(nval):
            fo = np.sort(forc[i,:])
            obs[i] = np.interp(pit1[i], ff, fo)

        pit2 = sutils.pit(obs, forc)

        self.assertTrue(np.allclose(pit1, pit2))


    def test_lhs(self):

        nparams = 10
        nsamples = 50
        pmin = 1
        pmax = 10

        samples = sutils.lhs(nparams, nsamples, pmin, pmax)

        for i in range(nparams):
            u = (np.sort(samples[:,i])-pmin)/(pmax-pmin)
            ff = (np.arange(1, nsamples+1)-0.3)/(nsamples+0.4)

            # Perform two sided KS test on results
            D = np.max(np.abs(u-ff))
            p = kolmogorov(D*math.sqrt(nsamples))

            self.assertTrue(p>0.95)

    def test_schaakeshuffle1(self):
        # Test from Clark et al. 2004

        obs = np.array([
            [10.7, 10.9, 13.5],
            [9.3, 9.1, 13.7],
            [6.8, 7.2, 9.3],
            [11.3, 10.7, 15.6],
            [12.2, 13.1, 17.8],
            [13.6, 14.2, 19.3],
            [8.9, 9.4, 12.1],
            [9.9, 9.2, 11.8],
            [11.8, 11.9, 15.2],
            [12.9, 12.5, 16.9]
        ])

        forc1 = np.array([
            [15.3, 9.3, 17.6],
            [11.2, 6.3, 15.6],
            [8.8, 7.9, 13.5],
            [11.9, 7.5, 14.2],
            [7.5, 13.5, 18.3],
            [9.7, 11.8, 15.9],
            [8.3, 8.6, 14.5],
            [12.5, 17.7, 23.9],
            [10.3, 7.2, 12.4],
            [10.1, 12.2, 16.3],
        ])


        expected1 = np.array([
            [10.1, 9.3, 14.5],
            [8.8, 7.2, 15.6],
            [7.5, 6.3, 12.4],
            [10.3, 8.6, 16.3],
            [11.9, 13.5, 18.3],
            [15.3, 17.7, 23.9],
            [8.3, 7.9, 14.2],
            [9.7, 7.5, 13.5],
            [11.2, 11.8, 15.9],
            [12.5, 12.2, 17.6]
        ])

        forc1_re = sutils.schaakeshuffle(obs, forc1)
        self.assertTrue(np.allclose(forc1_re, expected1))

        forc2 = np.concatenate([forc1, forc1, forc1], axis=0)
        forc2_re = sutils.schaakeshuffle(obs, forc2)

        expected2 = np.zeros((forc1.shape[0]*3, forc1.shape[1]))
        for i in range(expected2.shape[0]):
            k = int(i * float(forc1.shape[0])/expected2.shape[0])
            expected2[i,:] = expected1[k,:]

        self.assertTrue(np.allclose(forc2_re, expected2))


    def test_schaakeshuffle2(self):
        m1 = 0
        s1 = 10

        m2 = 20
        s2 = 20

        rho = 0.9

        nensA = 100
        nensB = 500

        # Generate first ensemble from multivariate
        # normal with high correlation
        cov = rho * math.sqrt(s1 * s2)
        ensA = np.random.multivariate_normal([m1, m2], [[s1, cov], [cov, s2]],
                size=nensA)

        # Generate second ensemble from two
        # independent normal variables
        # then reshuffled them
        ens1 = np.random.normal(m1, s1, size=nensB)
        ens2 = np.random.normal(m2, s2, size=nensB)
        ensB = np.array([ens1, ens2]).T
        sutils.schaakeshuffle(ensA, ensB, copy=False)

        # Check we are preserving rank correlation
        rA, pA = spearmanr(ensA[:,0], ensA[:,1])
        rB, pB = spearmanr(ensB[:,0], ensB[:,1])

        self.assertTrue(abs(rA-rB)<1e-4)


if __name__ == "__main__":
    unittest.main()
