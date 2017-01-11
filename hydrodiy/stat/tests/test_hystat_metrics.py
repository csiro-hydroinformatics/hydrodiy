import os
import unittest
import numpy as np
import pandas as pd

from scipy.stats import norm

from hydrodiy.stat import metrics
from hydrodiy.stat import transform, sutils

np.random.seed(0)

class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MetricsTestCase')
        source_file = os.path.abspath(__file__)
        ftest = os.path.dirname(source_file)

        fd1 = os.path.join(ftest, 'data', 'crps_testdata_01.txt')
        data = np.loadtxt(fd1)
        self.obs1 = data[:,0].copy()
        self.sim1 = data[:,1:].copy()

        frt1 = os.path.join(ftest, 'data', 'crps_testres_crpsmatens_01.txt')
        self.crps_reliabtab1 = np.loadtxt(frt1)

        fv1 = os.path.join(ftest, 'data', 'crps_testres_crpsvalens_01.txt')
        c1 = np.loadtxt(fv1)
        c1[2] *= -1
        self.crps_value1 = {
            'crps':c1[0],
            'reliability':c1[1],
            'resolution':c1[2],
            'uncertainty':c1[3],
            'potential':c1[4]
        }


        fd2 = os.path.join(ftest, 'data', 'crps_testdata_02.txt')
        data = np.loadtxt(fd2)
        self.obs2 = data[:,0].copy()
        self.sim2 = data[:,1:].copy()

        frt2 = os.path.join(ftest, 'data', 'crps_testres_crpsmatens_02.txt')
        self.crps_reliabtab2 = np.loadtxt(frt2)

        fv2 = os.path.join(ftest, 'data', 'crps_testres_crpsvalens_02.txt')
        c2 = np.loadtxt(fv2)
        c2[2] *= -1
        self.crps_value2 = {
            'crps':c2[0],
            'reliability':c2[1],
            'resolution':c2[2],
            'uncertainty':c2[3],
            'potential':c2[4]
        }


    def test_crps_reliability_table1(self):
        cr, rt = metrics.crps(self.obs1, self.sim1)
        for i in range(rt.shape[1]):
            self.assertTrue(np.allclose(rt.iloc[:, i], \
                self.crps_reliabtab1[:,i], atol=1e-5))

    def test_crps_reliability_table2(self):
        cr, rt = metrics.crps(self.obs2, self.sim2)
        for i in range(rt.shape[1]):
            self.assertTrue(np.allclose(rt.iloc[:, i], \
                self.crps_reliabtab2[:,i], atol=1e-5))

    def test_crps_value1(self):
        cr, rt = metrics.crps(self.obs1, self.sim1)
        for nm in cr.keys():
            ck = np.allclose(cr[nm], self.crps_value1[nm], atol=1e-5)
            self.assertTrue(ck)

    def test_crps_value2(self):
        cr, rt = metrics.crps(self.obs2, self.sim2)
        for nm in cr.keys():
            ck = np.allclose(cr[nm], self.crps_value2[nm], atol=1e-5)
            self.assertTrue(ck)

    def test_iqr(self):
        nforc = 100
        nens = 200

        ts = np.repeat(np.random.uniform(10, 20, nforc)[:, None], nens, 1)
        qq = sutils.ppos(nens)
        spread = 2*norm.ppf(qq)[None,:]

        ens = ts + spread
        # Double the spread. This should lead to iqr skill score of 33%
        # sk = (2*iqr-iqr)/(2*iqr+iqr) = 1./3
        ref = ts + spread*2

        iqr = metrics.iqr(ens, ref)
        expected = np.array([100./3, 2.67607, 5.35215])
        self.assertTrue(np.allclose(iqr, expected, atol=1e-4))


    def test_bias(self):
        obs = np.arange(0, 200)

        for name in ['Identity', 'Log', 'Reciprocal']:
            trans = metrics.get_transform(name)
            tobs = trans.forward(obs)
            tsim = tobs + 2
            sim = trans.backward(tsim)
            bias = metrics.bias(obs, sim, name)

            expected = 2./np.mean(tobs)
            self.assertTrue(np.allclose(bias, expected))


    def test_nse(self):
        obs = np.arange(0, 200)

        for name in ['Identity', 'Log', 'Reciprocal']:
            trans = metrics.get_transform(name)
            tobs = trans.forward(obs)
            tsim = tobs + 2
            sim = trans.backward(tsim)
            nse = metrics.nse(obs, sim, name)

            expected = 1-4.*len(obs)/np.sum((tobs-np.mean(tobs))**2)
            self.assertTrue(np.allclose(nse, expected))


if __name__ == "__main__":
    unittest.main()
