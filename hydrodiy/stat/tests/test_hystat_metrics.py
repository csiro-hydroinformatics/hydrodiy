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
        self.ftest = ftest

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


    def test_alpha(self):
        nval = 100
        nens = 500
        obs = np.linspace(0, 10, nval)
        sim = np.repeat(np.linspace(0, 10, nens)[:, None], \
                    nval, 1).T

        a, _ = metrics.alpha(obs, sim)
        self.assertTrue(np.allclose(a, 1.))


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


    def test_pit(self):
        nforc = 100
        nens = 200

        obs = np.linspace(0, 1, nforc)
        ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)

        p = metrics.pit(obs, ens)
        self.assertTrue(np.all(np.abs(obs-p)<8e-3))


    def test_cramer_von_mises(self):

        fd = os.path.join(self.ftest, 'data','cramer_von_mises_test_data.csv')
        data = pd.read_csv(fd, skiprows=15).values

        fe = os.path.join(self.ftest, \
                        'data','cramer_von_mises_test_data_expected.csv')
        expected = pd.read_csv(fe, skiprows=15)

        for nval in expected['nval'].unique():
            # Select data for nval
            exp = expected.loc[expected['nval'] == nval, :]

            for i in range(exp.shape[0]):
                x = data[i, :nval]

                st1 = exp['stat1'].iloc[i]
                pv1 = exp['pvalue1'].iloc[i]

                st2, pv2 = metrics.cramer_von_mises_test(x)

                ck1 = abs(st1-st2)<5e-3
                ck2 = abs(pv1-pv2)<1e-2

                self.assertTrue(ck1 and ck2)


    def test_alpha(self):
        nforc = 100
        nens = 200
        nrepeat = 50

        for i in range(nrepeat):
            obs = np.linspace(0, 1, nforc)
            ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)

            for type in ['CV', 'KS']:
                st, pv = metrics.alpha(obs, ens)
                self.assertTrue(pv>1.-1e-3)


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
        expected = np.array([100./3, 2.67607, 5.35215, 0.5])
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


    def test_nse_deterministic(self):
        ''' Testing regular deterministic NSE '''
        obs = np.arange(0, 200)+100.
        bias = 1.

        for name in ['Identity', 'Log', 'Reciprocal']:
            trans = metrics.get_transform(name)
            tobs = trans.forward(obs)
            tsim = tobs + bias
            sim = trans.backward(tsim)
            nse, accur, sharp = metrics.nse(obs, sim, name)

            expected = 1-bias**2*len(obs)/np.sum((tobs-np.mean(tobs))**2)
            self.assertTrue(np.allclose(nse, expected))
            self.assertTrue(np.allclose(accur, expected))
            self.assertTrue(np.allclose(sharp, 1.))


    def test_nse_probabilistic(self):
        ''' Testing probabilistic NSE '''
        nens = 10000
        nval = 200
        obs = np.linspace(0.1, 0.2, nval)

        bias = 1.
        sig = 0.5
        ff = sutils.ppos(nval)

        for name in ['Identity', 'Log', 'Reciprocal']:
            trans = metrics.get_transform(name)
            tobs = trans.forward(obs)
            ens = norm.ppf(ff)
            me = np.mean(ens)
            se = np.std(ens)
            ens = (ens-me)/se
            tsim = tobs[:, None]+bias+sig*ens[None, :]
            sim = trans.backward(tsim)
            nse, accur, sharp = metrics.nse(obs, sim, name)

            sso = np.sum((tobs-np.mean(tobs))**2)
            expected_accur = 1-nval*bias**2/sso
            expected_sharp = 1-nval*sig**2/sso
            expected_nse = 1-nval*(sig**2+bias**2)/sso

            ck = np.allclose(nse, expected_nse, atol=1e-3)
            self.assertTrue(ck)

            ck = np.allclose(accur, expected_accur, atol=1e-3)
            self.assertTrue(ck)

            ck = np.allclose(sharp, expected_sharp, atol=1e-3)
            self.assertTrue(ck)

            self.assertTrue(accur>nse)
            self.assertTrue(sharp>nse)


if __name__ == "__main__":
    unittest.main()
