import os
import unittest
import numpy as np
import pandas as pd

from hydrodiy.stat import metrics

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
            'crps_potential':c1[4]
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
            'crps_potential':c2[4]
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
        obs = np.arange(0, 200)
        sim = np.dot(np.arange(0, 100).reshape((100,1)), np.ones((1, 200))).T
        ref = np.array([obs]*200)
        iqr = metrics.iqr_scores(obs, sim, ref)
        expected = np.array([33.2446808, 50.2, 100.2])
        self.assertTrue(np.allclose(iqr, expected, atol=1e-2))

    def test_median_contingency(self):
        nens = 50
        obs = np.linspace(0.9, 1.1, 300)
        ref = np.array([obs]*len(obs))
        sim = np.vstack([np.random.uniform(1.1, 2., size=(100, nens)),
            np.random.uniform(0., 0.9, size=(200, nens))])

        cont, hit, miss, medians = metrics.median_contingency(obs, sim, ref)

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

        cont, hit, miss, hitlow, hithigh, terciles = metrics.tercile_contingency(obs, sim, ref)

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
        #import pdb; pdb.set_trace()

    def test_det_metrics(self):
        nval = 100
        nens = 1
        obs = pd.Series(np.random.normal(size=nval))
        sim = pd.DataFrame(np.random.normal(size=(nval,nens)))
        sc = metrics.det_metrics(obs, sim)
        sc, idx = metrics.det_metrics(obs, sim, True)

    def test_cut(self):
        nval = 10
        cats = np.array([-0.5, 0, 0.5])
        ysim1 = pd.Series(np.random.choice([-1, 0, 1], nval))

        returned = metrics.cut(ysim1, cats)

        expected = np.zeros((nval, 4))
        expected[ysim1.values==-1,0] = 1
        expected[ysim1.values==0,1] = 1
        expected[ysim1.values==1,3] = 1

        self.assertTrue(np.allclose(returned.values, expected, atol=1e-2))

        nens = 100
        cats = [0.5]
        ysim2 = pd.DataFrame(np.random.choice([0, 1], nval*nens).reshape((nval, nens)))
        returned = metrics.cut(ysim2, cats)
        m = ysim2.mean(axis=1)
        cc = returned.columns
        expected = pd.DataFrame({cc[0]:1-m, cc[1]:m})
        self.assertTrue(np.allclose(returned.values, expected, atol=1e-2))

    def test_drps(self):
        nval = 5
        nens = 20
        cats = np.ones((nval,1)) * 0.5
        yobs = pd.Series(np.random.choice([0, 1], nval))
        ysim = pd.DataFrame(np.random.choice([0, 1], nval*nens).reshape((nval, nens)))
        returned, drps_all = metrics.drps(yobs, ysim, cats)

        m = ysim.mean(axis=1)
        expected = (2*(m-yobs)**2).mean()
        self.assertTrue(np.allclose(returned, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
