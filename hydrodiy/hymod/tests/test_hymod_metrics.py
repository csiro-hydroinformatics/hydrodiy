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
        iqr = metrics.iqr_scores(obs, sim)
        returned = np.array([iqr['reliability'] , iqr['precision']])
        expected = np.array([0.5, 0.5])
        self.assertTrue(np.allclose(returned, expected, atol=1e-2))

    def test_ens_metrics(self):
        nval = 100
        nens = 50
        obs = pd.Series(np.random.normal(size=nval))
        sim = pd.DataFrame(np.random.normal(size=(nval,nens)))
        sc = metrics.ens_metrics(obs, sim)

    def test_det_metrics(self):
        nval = 100
        nens = 1
        obs = pd.Series(np.random.normal(size=nval))
        sim = pd.DataFrame(np.random.normal(size=(nval,nens)))
        sc = metrics.det_metrics(obs, sim)
        sc = metrics.det_metrics(obs, sim, True)

if __name__ == "__main__":
    unittest.main()
