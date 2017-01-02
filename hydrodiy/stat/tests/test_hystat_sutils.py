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
        fdata = '%s/data/acf1_data.csv'%self.ftest
        data, comment = csv.read_csv(fdata)
        data = data.astype(float)
        fres = '%s/data/acf1_result.csv'%self.ftest
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))
        self.assertTrue(np.allclose(res['acf'].values,
            expected['acf'].values))

    def test_acf2(self):
        fdata = '%s/data/acf2_data.csv'%self.ftest
        data, comment = csv.read_csv(fdata)
        fres = '%s/data/acf2_result.csv'%self.ftest
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))

        self.assertTrue(np.allclose(res['acf'].values,
            expected['acf'].values))

    def test_acf3(self):
        fdata = '%s/data/acf1_data.csv'%self.ftest
        data, comment = csv.read_csv(fdata)
        data = data.astype(float).values
        data = np.vstack([data[:5], 1000.,  np.nan, data[5:],
                    np.nan, np.nan])

        fres = '%s/data/acf1_result.csv'%self.ftest
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
            ff = sutils.empfreq(nsamples)

            # Perform two sided KS test on results
            D = np.max(np.abs(u-ff))
            p = kolmogorov(D*math.sqrt(nsamples))

            self.assertTrue(p>0.95)


if __name__ == "__main__":
    unittest.main()
