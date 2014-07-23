import os
import math
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2

from hyio import csv
from hystat.bootstrap import Bootstrap

class BootstrapTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BootstrapTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

        def fun(nval, mu, sig):
            return np.random.normal(size=nval, loc=mu, scale=sig)
        self.getsample = fun
       
    def test_print(self):
        x = np.random.normal(size=100)
        boot = Bootstrap(x)
        print(boot)

    def test_ci_percent_normalmean(self):
        nval = 500
        sig = 10.
        x = self.getsample(nval=nval, mu=10, sig=sig)
        boot = Bootstrap(x, lambda x: np.mean(x))
        ci = boot.get_ci_percent()

        ci2 = np.empty((1,3), float)
        ci2[0,0] = np.mean(x)
        ci2[0,1:] = norm.ppf([0.025, 0.975], 
                loc=np.mean(x), scale=sig/math.sqrt(nval))
        ck = np.allclose(ci, ci2, rtol=1e-1)
        self.assertTrue(ck)

    def test_ci_bca_normalmean(self):
        nval = 500
        sig = 10.
        x = self.getsample(nval=nval, mu=10, sig=sig)
        boot = Bootstrap(x, lambda x: np.mean(x))
        ci = boot.get_ci_bca()

        ci2 = np.empty((1,3), float)
        ci2[0,0] = np.mean(x)
        ci2[0,1:] = norm.ppf([0.025, 0.975], 
                loc=np.mean(x), scale=sig/math.sqrt(nval))
        ck = np.allclose(ci, ci2, rtol=1e-1)
        self.assertTrue(ck)

    def test_ci_percent_normalmeanvar(self):
        nval = 500
        sig = 10.
        x = self.getsample(nval=nval, mu=10, sig=sig)
        fact = math.sqrt((nval+0.)/(nval-1.))
        boot = Bootstrap(x, 
                    nboot = 5000,
                    statistic=lambda u: [np.mean(u), np.std(u)*fact])
        ci = boot.get_ci_percent()

        ci2 = np.empty((2,3), float)
        ci2[0,0] = np.mean(x)
        ci2[0,1:] = norm.ppf([0.025, 0.975], 
                loc=np.mean(x), scale=sig/math.sqrt(nval))
        ci2[1,0] = np.std(x)
        ci2[1,1:] = sig/(nval-1)*chi2.ppf([0.025, 0.975], df=nval-1) 

        ck = np.allclose(ci, ci2, atol=1e0)
        self.assertTrue(ck)

    def test_ci_efron(self):
        f = '%s/efron_tibshirani_chap14.csv'%self.FOUT
        data, comment = csv.read_csv(f)
        data = data.T
        boot = Bootstrap(data[0], lambda x: np.var(x), 2000)

        ci1 = boot.get_ci_percent()
        ci2 = boot.get_ci_bca()

        # Check BCa parameters
        ck_a = np.allclose(boot.a[0], 0.061, atol=1e-2)
        self.assertTrue(ck_a)

        ck_z0 = np.allclose(boot.z0[0], 0.146, atol=1e-1)
        self.assertTrue(ck_z0)

if __name__ == "__main__":
    unittest.main()
