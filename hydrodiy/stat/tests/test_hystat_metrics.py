import os, re, math
import unittest
import warnings
from itertools import product as prod

import time
import zipfile

import matplotlib as mpl
mpl.use('Agg')
print('Using {} backend'.format(mpl.get_backend()))

import matplotlib.pyplot as plt

from scipy.stats import norm, lognorm, spearmanr
import numpy as np
import pandas as pd

from hydrodiy.stat import metrics
from hydrodiy.io import csv
from hydrodiy.stat import transform, sutils
from hydrodiy.stat.censored import normcensfit2d

from hydrodiy.plot import putils

# Try to import C code
HAS_C_STAT_MODULE = True
try:
    import c_hydrodiy_stat
except ImportError:
    HAS_C_STAT_MODULE = False

from vrf_scores import crps_ecdf as crps_csiro

np.random.seed(0)

class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MetricsTestCase')
        source_file = os.path.abspath(__file__)
        ftest = os.path.dirname(source_file)
        self.ftest = ftest
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

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

        a, _, _ = metrics.alpha(obs, sim)
        self.assertTrue(np.allclose(a, 1.))


    def test_crps_csiro(self):
        ''' Compare CRPS calculation with reference code '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 100
        nens = 1000
        nrepeat = 100
        sparam = 1
        sign = 2

        for irepeat in range(nrepeat):
            obs = lognorm.rvs(size=nval, s=sparam, loc=0, scale=1)
            noise = norm.rvs(size=(nval, nens), scale=sign)
            trend = norm.rvs(size=nval, scale=sign*1.5)
            ens = obs[:, None]+noise+trend[:, None]

            cr, _ = metrics.crps(obs, ens)

            # Reference computation
            ccr = [crps_csiro(forc, o) for forc, o in zip(ens, obs)]
            self.assertTrue(np.isclose(cr.crps, np.mean(ccr)))


    def test_crps_reliability_table1(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        cr, rt = metrics.crps(self.obs1, self.sim1)
        for i in range(rt.shape[1]):
            self.assertTrue(np.allclose(rt.iloc[:, i], \
                self.crps_reliabtab1[:,i], atol=1e-5))


    def test_crps_reliability_table2(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        cr, rt = metrics.crps(self.obs2, self.sim2)
        for i in range(rt.shape[1]):
            self.assertTrue(np.allclose(rt.iloc[:, i], \
                self.crps_reliabtab2[:,i], atol=1e-5))


    def test_crps_value1(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        cr, rt = metrics.crps(self.obs1, self.sim1)
        for nm in cr.keys():
            ck = np.allclose(cr[nm], self.crps_value1[nm], atol=1e-5)
            self.assertTrue(ck)


    def test_crps_value2(self):
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        pit, sudo = metrics.pit(obs, ens)

        self.assertTrue(np.all(np.abs(obs-pit)<8e-3))
        self.assertTrue(np.all(~sudo[1:]))
        self.assertTrue(sudo[0])


    def test_pit_hassan(self):
        ''' Test pit as per Hassan requirement '''
        obs = [3]
        ens = [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]

        # using scipy func
        pit, sudo = metrics.pit(obs, ens, random=False)
        self.assertTrue(np.isclose(pit, 0.5666666666666))
        self.assertTrue(np.all(~sudo))

        # using randomisation
        nrepeat = 1000
        pits = np.array([metrics.pit(obs, ens, random=True)[0] \
                    for i in range(nrepeat)]).squeeze()
        pits = pd.Series(pits).value_counts().sort_index()
        self.assertTrue(np.allclose(pits.index, [0.33121, 0.394904, \
                        0.458599, 0.522293, 0.585987, 0.649682, 0.713376]))
        self.assertTrue((pits>100).all())


    def test_pit_hassan_rpp(self):
        ''' Test pit from Hassan's RPP data '''

        frpp = os.path.join(self.ftest, 'data', 'rpp_data.zip')
        with zipfile.ZipFile(frpp, 'r') as archive:
            obs, _ = csv.read_csv('obs.csv', archive=archive, index_col=0, \
                                                has_colnames=False)
            ens, _ = csv.read_csv('rpp_ensemble.csv', archive=archive, \
                                            index_col=0)
        pit1, sudo1 = metrics.pit(obs, ens, random=False)
        pit2, sudo2 = metrics.pit(obs, ens, random=True)

        plt.close('all')
        try:
            fig, ax = plt.subplots()
        except:
            self.skipTest('Cannot initialise matplotlib, not too sure why')

        ff = sutils.ppos(len(pit1))

        kk = np.argsort(pit1)
        p1 = pit1[kk]
        s1 = sudo1[kk]
        ax.plot(p1[s1], ff[s1], 'k.', \
                    markersize=5, alpha=0.3, \
                    label='Scipy percentileofscore (sudo)')
        ax.plot(p1[~s1], ff[~s1], 'o',
                    markeredgecolor='k', \
                    markerfacecolor='k', \
                    label='Scipy percentileofscore')


        kk = np.argsort(pit2)
        p2 = pit2[kk]
        s2 = sudo2[kk]
        ax.plot(p2[s2], ff[s2], 'r.', \
                    markersize=5, alpha=0.5, \
                    label='hydrodiy.metrics.pit using random=True (sudo)')

        ax.plot(p2[~s2], ff[~s2], 'o', \
                    markeredgecolor='r', \
                    markerfacecolor='r', \
                    label='hydrodiy.metrics.pit using random=True')

        ax.legend(loc=2, framealpha=0.5)
        ax.set_xlabel('PIT [-]')
        ax.set_ylabel('ECDF [-]')

        fp = os.path.join(self.fimg, 'pit_hassan.png')
        fig.savefig(fp)


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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
                st, pv, sudo = metrics.alpha(obs, ens)
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
            ck = np.isclose(bias, expected)
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
            self.assertTrue(np.isclose(nse, expected))


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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

        nval = 10
        nens = 100
        obs = np.arange(nval)

        sim = obs[:, None] + np.random.uniform(-1e-3, 1e-3, size=(nval, nens))
        D = metrics.dscore(obs, sim)
        self.assertTrue(np.allclose(D, 1.))

        sim = -obs[:, None] + np.random.uniform(-1e-3, 1e-3, \
                                            size=(nval, nens))
        D = metrics.dscore(obs, sim)
        self.assertTrue(np.allclose(D, 0.))


    def test_ensrank_large_ensemble(self):
        ''' Test ensrank for large ensemble numbers '''
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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
        if not HAS_C_STAT_MODULE:
            self.skipTest('Missing C module c_hydrodiy_stat')

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


    def test_kge(self):
        ''' Testing  KGE '''
        obs = np.arange(0, 200)+100.
        bias = 0.1

        for trans in self.transforms:
            if trans.params.nval > 0:
                trans.params.values[0] = np.mean(obs)*1e-2

            # First trial - multiplicative bias
            tobs = trans.forward(obs)
            tsim = tobs*(1+bias)
            sim = trans.backward(tsim)
            kge = metrics.kge(obs, sim, trans)

            expected = 1-math.sqrt(2)*bias
            self.assertTrue(np.isclose(kge, expected))

            # Second trial - additive bias
            tsim = tobs - bias
            sim = trans.backward(tsim)
            kge = metrics.kge(obs, sim, trans)

            expected = 1-bias/abs(np.mean(tobs))
            self.assertTrue(np.isclose(kge, expected))

            # Third trial - random error
            tsim = tobs + 1e-2*np.mean(tobs)*np.random.uniform(-1, 1, \
                                                        size=len(tobs))
            sim = trans.backward(tsim)
            kge = metrics.kge(obs, sim, trans)

            bias = np.mean(tsim)/np.mean(tobs)
            rstd = np.std(tsim)/np.std(tobs)
            corr = np.corrcoef(tobs, tsim)[0, 1]
            expected = 1-math.sqrt((1-bias)**2+(1-rstd)**2+(1-corr)**2)
            self.assertTrue(np.isclose(kge, expected))


    def test_kge_warnings(self):
        ''' Testing KGE warnings '''

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            obs = np.zeros(100)
            sim = np.random.uniform(0, 1, size=100)
            try:
                kge = metrics.kge(obs, sim)
            except Warning as warn:
                self.assertTrue(str(warn).startswith('KGE - Mean value'))
            else:
                raise ValueError('Problem in error handling')

            obs = np.ones(100)
            sim = np.random.uniform(0, 1, size=100)
            try:
                kge = metrics.kge(obs, sim)
            except Warning as warn:
                self.assertTrue(str(warn).startswith('KGE - Standard dev'))
            else:
                raise ValueError('Problem in error handling')

            obs = np.random.uniform(0, 1, size=100)
            sim = np.ones(100)
            try:
                kge = metrics.kge(obs, sim)
            except Warning as warn:
                self.assertTrue(str(warn).startswith('KGE - Standard dev'))
            else:
                raise ValueError('Problem in error handling')


    def test_corr2d(self):
        ''' Test correlation for ensemble data '''
        nval = 200
        nens = 1000
        obs = np.arange(0, nval).astype(float)

        for trans, type, stat in prod(self.transforms, \
                    ['Pearson', 'Spearman', 'censored'], \
                    ['mean', 'median']):

            if trans.params.nval > 0:
                trans.params.values[0] = np.mean(obs)*1e-2

            tobs = trans.forward(obs)
            tens = tobs[:, None] - 2 \
                        + np.random.uniform(-2, 2, size=(nval, nens))
            ens = trans.backward(tens)
            corr = metrics.corr(obs, ens, trans, False, stat, type)

            if stat == 'mean':
                tsim = np.nanmean(tens, 1)
            else:
                tsim = np.nanmedian(tens, 1)

            if type == 'Pearson':
                expected = np.corrcoef(tobs, tsim)[0, 1]
            elif type == 'Spearman':
                expected = spearmanr(tobs, tsim).correlation
            else:
                X = np.column_stack([tobs, tsim])
                _, _, expected = normcensfit2d(X, censor=1e-10)

            ck = np.isclose(corr, expected)
            self.assertTrue(ck)


    def test_corr1d(self):
        ''' Test correlation  for deterministic data '''
        nval = 200
        obs = np.arange(0, nval).astype(float)

        for trans, type, stat in prod(self.transforms, \
                    ['Pearson', 'Spearman'], ['mean', 'median']):

            if trans.params.nval > 0:
                trans.params.values[0] = np.mean(obs)*1e-2

            tobs = trans.forward(obs)
            tens = tobs - 2 \
                        + np.random.uniform(-1, 1, size=nval)
            ens = trans.backward(tens)
            corr = metrics.corr(obs, ens, trans, False, stat, type)

            if type == 'Pearson':
                expected = np.corrcoef(tobs, tens)[0, 1]
            elif type == 'Spearman':
                expected = spearmanr(tobs, tens).correlation
            else:
                X = np.column_stack([tobs, tens])
                _, _, expected = normcensfit2d(X, censor=1e-10)

            ck = np.isclose(corr, expected)
            self.assertTrue(ck)


    def test_abspeakerror(self):
        ''' Test peak timing error using lagged data '''
        nval = 2000
        obs = np.exp(np.random.normal(size=nval))

        lag = 3
        sim = np.append(obs[lag:], [0]*2)
        aperr, events = metrics.absolute_peak_error(obs, sim)

        self.assertTrue(np.isclose(aperr, lag))
        self.assertTrue(np.allclose(events.delta, lag))


    def test_relpercerror(self):
        ''' Test relative percentile error '''
        nval = 2000
        obs = np.exp(np.random.normal(size=nval))
        err = 1.3
        sim = (err+1)*obs

        rperr, perc = metrics.relative_percentile_error(obs, sim, [0, 100])
        self.assertTrue(np.isclose(rperr, err))
        self.assertTrue(np.allclose(perc.rel_perc_err, err))

        rperr, perc = metrics.relative_percentile_error(obs, sim, [0, 100], \
                                                modified=True)
        errm = err/(2+err)
        self.assertTrue(np.isclose(rperr, abs(errm)))
        self.assertTrue(np.allclose(perc.rel_perc_err, errm))


    def test_binary(self):
        ''' Test binary metrics '''
        # Generating Finley forecasts table
        # using dummy variables
        # see Stephenson, David B. "Use of the odds ratio for diagnosing
        #   forecast skill." Weather and Forecasting 15.2 (2000): 221-232.
        mat = [[28, 72], [23, 2680]]
        scores = metrics.binary(mat)

        # tests
        self.assertEqual(scores['truepos'], 28)
        self.assertEqual(scores['trueneg'], 2680)
        self.assertEqual(scores['falsepos'], 72)
        self.assertEqual(scores['falseneg'], 23)

        # See Table 5 in Stephenson, 2000
        self.assertTrue(np.isclose(scores['bias'], 1.96, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['hitrate'], 0.549, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['falsealarm'], 0.026, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['oddsratio'], 45.314, rtol=0, atol=1e-3))
        # .. not exactly the value reported by Stepenson due to rounding
        self.assertTrue(np.isclose(scores['hitrate_random'], 0.035, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['falsealarm_random'], 0.036, rtol=0, atol=1e-3))

        # See Table 7 in Stephenson, 2000
        #.. corresponds to the square root of the Pearson
        self.assertTrue(np.isclose(scores['MCC'], math.sqrt(0.142), rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['MCC_random'], 0.0, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['accuracy'], 0.966, rtol=0, atol=1e-3))
        self.assertTrue(np.isclose(scores['accuracy_random'], 0.948, rtol=0, atol=1e-3))

        # Additional scores
        self.assertTrue(np.isclose(scores['precision'], 0.28, rtol=0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
