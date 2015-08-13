import os
import unittest
import numpy as np
from hyio import csv
from hystat import sutils

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
        xq = sutils.percentiles(x, qq, 1.)
        self.assertTrue(np.allclose(xq*100, qq))

    def test_acf1(self):
        fdata = '%s/acf1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        data = data.astype(float)
        fres = '%s/acf1_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))
        self.assertTrue(np.allclose(res['acf'].values, 
            expected['acf'].values))

    def test_acf2(self):
        fdata = '%s/acf2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        fres = '%s/acf2_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        res = sutils.acf(data, lag=range(0,6))

        self.assertTrue(np.allclose(res['acf'].values, 
            expected['acf'].values))

    def test_acf3(self):
        fdata = '%s/acf1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fdata)
        data = data.astype(float).values
        data = np.vstack([data[:5], 1000.,  np.nan, data[5:], 
                    np.nan, np.nan])

        fres = '%s/acf1_result.csv'%self.FOUT
        expected, comment = csv.read_csv(fres)

        filter = np.array([True]*len(data))
        filter[5] = False

        res = sutils.acf(data, lag=range(0,6), filter=filter)
        self.assertTrue(np.prod(np.abs(res['acf'])<=1+1e-10)==1)

    def test_ar1(self):
        nval = 10
        params = np.array([0.9, 10., 5.])
        y = sutils.ar1random(params, nval)
        innov = sutils.ar1inverse(params[:2], y)
        y2 = sutils.ar1innov(params[:2], innov)
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
  

if __name__ == "__main__":
    unittest.main()
