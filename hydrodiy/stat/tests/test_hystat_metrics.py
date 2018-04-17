import os, re
import unittest
import numpy as np
import pandas as pd

from scipy.stats import norm
import time

from hydrodiy.stat import metrics
from hydrodiy.stat import transform, sutils

import c_hydrodiy_stat

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

        self.transforms = [
            transform.Identity(),
            transform.Log(),
            transform.Reciprocal()
        ]


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
        ''' Test pit computation '''
        nforc = 100
        nens = 200

        obs = np.linspace(0, 1, nforc)
        ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)

        p = metrics.pit(obs, ens)
        self.assertTrue(np.all(np.abs(obs-p)<8e-3))


    def test_cramer_von_mises(self):
        ''' test Cramer Von-Mises test '''

        fd = os.path.join(self.ftest, 'data',\
                    'cramer_von_mises_test_data.csv')
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


    def test_cramer_von_mises2(self):
        ''' Second test of Cramer Von Mises test '''

        fd = os.path.join(self.ftest, 'data',\
                        'testdata_AD_CVM.csv')
        data = pd.read_csv(fd, skiprows=15)
        cc = [cn for cn in data.columns if re.search('^x', cn)]

        for _, row in data.iterrows():
            unifdata = row[cc]
            st1 = row['CVM_stat']
            pv1 = row['CVM_pvalue']
            st2, pv2 = metrics.cramer_von_mises_test(unifdata)
            ck = np.allclose([st1, pv1], [st2, pv2], atol=1e-3)
            self.assertTrue(ck)


    def test_anderson_darling(self):
        ''' test Anderson Darling test '''

        fd = os.path.join(self.ftest, 'data',\
                        'testdata_AD_CVM.csv')
        data = pd.read_csv(fd, skiprows=15)
        cc = [cn for cn in data.columns if re.search('^x', cn)]

        for _, row in data.iterrows():
            unifdata = row[cc]
            st1 = row['AD_stat']
            pv1 = row['AD_pvalue']
            st2, pv2 = metrics.anderson_darling_test(unifdata)

            ck = np.allclose([st1, pv1], [st2, pv2])
            self.assertTrue(ck)


    def test_anderson_darling_error(self):
        ''' test Anderson Darling test errors '''

        nval = 20
        unifdata = np.random.uniform(0, 1,  size=nval)
        unifdata[-1] = 10
        try:
            st, pv = metrics.anderson_darling_test(unifdata)
        except ValueError as err:
            self.assertTrue(str(err).startswith('ad_test'))
        else:
            raise ValueError('Problem in error handling')


    def test_alpha(self):
        nforc = 100
        nens = 200
        nrepeat = 50

        for i in range(nrepeat):
            obs = np.linspace(0, 1, nforc)
            ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)

            for type in ['CV', 'KS', 'AD']:
                st, pv = metrics.alpha(obs, ens)
                self.assertTrue(pv>1.-1e-3)


    def test_iqr(self):
        ''' Testing IQR for normally distributed forecasts '''
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


    def test_iqr_error(self):
        ''' Testing IQR error '''
        ens = np.random.uniform(0, 1, (100, 50))
        ref = np.random.uniform(0, 1, (80, 50))
        try:
            iqr = metrics.iqr(ens, ref)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected clim'))
        else:
            raise ValueError('Problem with error handling')


    def test_bias(self):
        ''' Test bias '''
        obs = np.arange(0, 200).astype(float)

        for trans in self.transforms:
            if trans.params.nval > 0:
                trans.params.values[0] = np.mean(obs)*1e-2

            tobs = trans.forward(obs)
            tsim = tobs - 2
            sim = trans.backward(tsim)
            bias = metrics.bias(obs, sim, trans)

            expected = -2./np.mean(tobs)
            ck = np.allclose(bias, expected)
            self.assertTrue(ck)


    def test_bias_error(self):
        ''' Test bias error '''
        obs = np.arange(0, 200)
        sim = np.arange(0, 190)
        try:
            bias = metrics.bias(obs, sim)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected sim'))
        else:
            raise ValueError('Problem with error handling')


    def test_nse(self):
        ''' Testing  NSE '''
        obs = np.arange(0, 200)+100.
        bias = -1.

        for trans in self.transforms:
            if trans.params.nval > 0:
                trans.params.values[0] = np.mean(obs)*1e-2

            tobs = trans.forward(obs)
            tsim = tobs + bias
            sim = trans.backward(tsim)
            nse = metrics.nse(obs, sim, trans)

            expected = 1-bias**2*len(obs)/np.sum((tobs-np.mean(tobs))**2)
            self.assertTrue(np.allclose(nse, expected))


    def test_nse_error(self):
        ''' Test bias error '''
        obs = np.arange(0, 200)
        sim = np.arange(0, 190)
        try:
            bias = metrics.nse(obs, sim)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected sim'))
        else:
            raise ValueError('Problem with error handling')


    def test_ensrank_weigel_data(self):
        ''' Testing ensrank C function  against data from
        Weigel and Mason (2011) '''

        sim = np.array([[22, 23, 26, 27, 32], \
            [28, 31, 33, 34, 36], \
            [24, 25, 26, 27, 28]], dtype=np.float64)

        fmat = np.zeros((3, 3), dtype=np.float64)
        ranks = np.zeros(3, dtype=np.float64)
        eps = np.float64(1e-6)

        c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

        fmat_expected = np.array([\
                [0., 0.08, 0.44], \
                [0., 0., 0.98], \
                [0., 0., 0.]])
        self.assertTrue(np.allclose(fmat, fmat_expected))

        ranks_expected = [1., 3., 2.]
        self.assertTrue(np.allclose(ranks, ranks_expected))


    def test_ensrank_deterministic(self):
        ''' Testing ensrank C function for deterministic simulations '''

        nval = 5
        nrepeat = 100

        for i in range(nrepeat):
            sim = np.random.uniform(0, 1, (nval, 1))
            fmat = np.zeros((nval, nval), dtype=np.float64)
            ranks = np.zeros(nval, dtype=np.float64)
            eps = np.float64(1e-6)

            c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

            # Zero on the diagonal
            self.assertTrue(np.allclose(np.diag(fmat), np.zeros(nval)))

            # Correct rank
            ranks_expected = 1+np.argsort(np.argsort(sim[:, 0]))

            xx, yy = np.meshgrid(sim[:, 0], sim[:, 0])
            tmp = (xx>yy).astype(float).T
            fmat_expected = np.zeros((nval, nval))
            idx = np.triu_indices(nval)
            fmat_expected[idx] = tmp[idx]

            self.assertTrue(np.allclose(ranks, ranks_expected))
            self.assertTrue(np.allclose(fmat, fmat_expected))


    def test_ensrank_python(self):
        ''' Test ensrank against python code '''
        nval = 4
        nens = 5
        nrepeat = 100
        eps = np.float64(1e-6)

        for irepeat in range(nrepeat):
            for ties in [True, False]:
                if ties:
                    sim = np.round(np.random.uniform(0, 100, (nval, nens))/10)
                else:
                    sim = np.random.uniform(0, 100, (nval, nens))

                fmat = np.zeros((nval, nval), dtype=np.float64)
                ranks = np.zeros(nval, dtype=np.float64)
                c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

                # Run python
                fmat_expected = fmat * 0.
                for i in range(nval):
                    for j in range(i+1, nval):
                        # Basic rankings
                        comb = np.concatenate([sim[i], sim[j]])
                        rk = np.argsort(np.argsort(comb))+1.

                        # Handle ties
                        for cu in np.unique(comb):
                            idx = np.abs(comb-cu)<eps
                            if np.sum(idx)>0:
                                rk[idx] = np.mean(rk[idx])

                        # Compute rank sums
                        srk = np.sum(rk[:nens])
                        fm = (srk-(nens+1.)*nens/2)/nens/nens;
                        fmat_expected[i, j] = fm

                # Ranks
                F = fmat_expected.copy()
                idx = np.tril_indices(nval)
                F[idx] = 1.-fmat_expected.T[idx]
                c1 = np.sum((F>0.5).astype(int), axis=1)
                c2 = np.sum(((F>0.5-1e-8) & (F<0.5+1e-8)).astype(int), axis=1)
                ranks_expected = c1+0.5*c2

                ck = np.allclose(fmat, fmat_expected)
                self.assertTrue(ck)

                ck = np.allclose(ranks, ranks_expected)
                self.assertTrue(ck)


    def test_dscore_perfect(self):
        ''' Test dscore for perfect correlation '''
        nval = 10
        nens = 100
        obs = np.arange(nval)

        sim = obs[:, None] + np.random.uniform(-1e-3, 1e-3, size=(nval, nens))
        D = metrics.dscore(obs, sim)
        self.assertTrue(np.allclose(D, 1.))

        sim = -obs[:, None] + np.random.uniform(-1e-3, 1e-3, size=(nval, nens))
        D = metrics.dscore(obs, sim)
        self.assertTrue(np.allclose(D, 0.))


    def test_ensrank_large_ensemble(self):
        ''' Test ensrank for large ensemble numbers '''

        nval = 50
        nens = 5000
        fmat = np.zeros((nval, nval), dtype=np.float64)
        ranks = np.zeros(nval, dtype=np.float64)
        sim = np.random.uniform(0, 1, (nval, nens))
        eps = np.float64(1e-6)

        t0 = time.time()
        c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)
        t1 = time.time()

        # Max 30 sec to compute this
        self.assertTrue(t1-t0<30)


    def test_ensrank_long_timeseries(self):
        ''' Test ensrank for long time series '''
        nval = 1000
        nens = 100
        fmat = np.zeros((nval, nval), dtype=np.float64)
        ranks = np.zeros(nval, dtype=np.float64)
        sim = np.random.uniform(0, 1, (nval, nens))
        eps = np.float64(1e-6)

        t0 = time.time()
        c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)
        t1 = time.time()

        # Max 3 sec to compute this
        self.assertTrue(t1-t0<100)


if __name__ == "__main__":
    unittest.main()
