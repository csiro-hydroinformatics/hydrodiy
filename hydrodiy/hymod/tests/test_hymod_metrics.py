import os
import unittest
import numpy as np
import pandas as pd
from hymod import metrics

class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> MetricsTestCase')
        FTEST, testfile = os.path.split(__file__)

        fd1 = os.path.join(FTEST, 'data', 'crps_testdata_01.txt')
        data = np.loadtxt(fd1)
        self.obs1 = data[:,0].copy() 
        self.sim1 = data[:,1:].copy() 
        frt1 = os.path.join(FTEST, 'data', 'crps_testres_crpsmatens_01.txt')
        self.crps_reliabtab1 = np.loadtxt(frt1) 
        fv1 = os.path.join(FTEST, 'data', 'crps_testres_crpsvalens_01.txt')
        self.crps_value1 = np.loadtxt(fv1) 
        self.crps_value1[2] *= -1
        

        fd2 = os.path.join(FTEST, 'data', 'crps_testdata_02.txt')
        data = np.loadtxt(fd2)
        self.obs2 = data[:,0].copy() 
        self.sim2 = data[:,1:].copy() 
        frt2 = os.path.join(FTEST, 'data', 'crps_testres_crpsmatens_02.txt')
        self.crps_reliabtab2 = np.loadtxt(frt2) 
        fv2 = os.path.join(FTEST, 'data', 'crps_testres_crpsvalens_02.txt')
        self.crps_value2 = np.loadtxt(fv2) 
        self.crps_value2[2] *= -1

    def test_crps_reliability_table1(self):
        cr, rt = metrics.crps(self.obs1, self.sim1)
        self.crps_reliabtab1.dtype = rt.dtype
        for nm in rt.dtype.names:
            self.assertTrue(np.allclose(rt[nm], self.crps_reliabtab1[nm], atol=1e-5))

    def test_crps_reliability_table2(self):
        cr, rt = metrics.crps(self.obs2, self.sim2)
        self.crps_reliabtab2.dtype = rt.dtype
        for nm in rt.dtype.names:
            self.assertTrue(np.allclose(rt[nm], self.crps_reliabtab2[nm], atol=1e-5))

    def test_crps_value1(self):
        cr, rt = metrics.crps(self.obs1, self.sim1)
        self.crps_value1.dtype = cr.dtype
        for nm in cr.dtype.names:
            self.assertTrue(np.allclose(cr[nm], self.crps_value1[nm], atol=1e-5))

    def test_crps_value2(self):
        cr, rt = metrics.crps(self.obs2, self.sim2)
        self.crps_value2.dtype = cr.dtype
        for nm in cr.dtype.names:
            self.assertTrue(np.allclose(cr[nm], self.crps_value2[nm], atol=1e-5))

    def test_iqr(self):
        obs = np.arange(0, 200)
        sim = np.dot(np.arange(0, 100).reshape((100,1)), np.ones((1, 200))).T
        ref = np.array([obs]*200) 
        out, iqr, iqr_clim, rel, rel_clim = metrics.iqr_scores(obs, sim, ref)
        returned = np.array([out['reliability_score'] , out['precision_score'],
                            out['reliability_skill']/100 , out['precision_skill']/100])
        expected = np.array([0.25, 0.5, 0.5, 0.5])
        self.assertTrue(np.allclose(returned, expected, atol=1e-2))

    def test_median_contingency(self):
        nens = 50
        obs = np.linspace(0.9, 1.1, 300)
        ref = np.array([obs]*len(obs))
        sim = np.vstack([np.random.uniform(1.1, 2., size=(100, nens)),
            np.random.uniform(0., 0.9, size=(200, nens))])

        cont, hit, miss, obs_med = metrics.median_contingency(obs, sim, ref)

        # check balance of cont table
        returned = (np.sum(cont[0,:]) - np.sum(cont[1,:]))/np.sum(cont)
        expected = 0.
        self.assertTrue(np.allclose(returned, expected, atol=1e-2))

        # Check hit and miss
        returned = np.array([hit, miss]) 
        expected = np.array([1./6, 0.75])
        self.assertTrue(np.allclose(returned, expected, atol=1e-7))

    def test_tercile_contingency(self):
        nens = 50
        obs = np.arange(0, 301)
        ref = np.array([obs]*len(obs))
        sim = np.vstack([np.random.uniform(1., 99., size=(200, nens)),
            np.random.uniform(200., 300., size=(101, nens))])
    
        cont, hit, miss, hitlow, hithigh, obs_t1, obs_t2 = metrics.tercile_contingency(obs, sim, ref)
 
        # check balance of cont table
        returned = (abs(np.sum(cont[0,:]) - np.sum(cont[1,:])) + abs(np.sum(cont[0,:]) - np.sum(cont[2,:])))/np.sum(cont)
        expected = 0.
        self.assertTrue(np.allclose(returned, expected, atol=1e-2))

        # Check hit / miss
        returned = np.array([hit, miss, hitlow, hithigh]) 
        expected = np.array([2./3, 0.5, 0.5, 1.])
        self.assertTrue(np.allclose(returned, expected, atol=1e-2))

    def test_ens_metrics(self):
        nval = 100
        nens = 50
        obs = pd.Series(np.random.normal(size=nval))
        ref = np.array([np.random.choice(obs.values, nens+10)]*nval)
        sim = pd.DataFrame(np.random.normal(size=(nval,nens)))
        sc, idx, rt, cont_med, cont_terc = metrics.ens_metrics(obs, sim, ref)

    def test_det_metrics(self):
        nval = 100
        nens = 1
        obs = pd.Series(np.random.normal(size=nval))
        sim = pd.DataFrame(np.random.normal(size=(nval,nens)))
        sc = metrics.det_metrics(obs, sim)
        sc, idx = metrics.det_metrics(obs, sim, True)

if __name__ == "__main__":
    unittest.main()
