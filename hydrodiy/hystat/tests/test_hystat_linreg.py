import os
import math
import unittest

import numpy as np
import pandas as pd

from hyio import csv
from hystat import linreg
from hystat import sutils

class LinregTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> LinregTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        self.y1 = np.array(1)
       
    def test_print(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]
        lm = linreg.Linreg(x, y)
        print(lm)

    def test_ols_johnston(self):
        # data set from Johnston and Di Nardo, page 75
        # Econometrics Methods, 1993
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        self.assertTrue(np.allclose(lm.params['estimate'].values, 
                                np.array([4., 2.5, -1.5])))

        sigma = lm.sigma
        self.assertTrue(sigma==math.sqrt(0.75))

        ck = np.allclose(lm.params['stderr'][2], sigma*math.sqrt(2.5))
        self.assertTrue(ck)

        ci = lm.params[['confint_025', 'confint_975']][-1:]
        ck = np.allclose(ci, [-7.39, 4.39], atol=1e-2)
        self.assertTrue(ck)

        y0, pint = lm.predict(np.array([10, 10]).reshape((1,2)))
        self.assertTrue(np.allclose(y0[0],14))
 
    def test_ols_R1(self):
        # data set from R
        fd = '%s/linreg1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)
        fd = '%s/linreg1_result_estimate.csv'%self.FOUT
        estimate, comment = csv.read_csv(fd)
        fd = '%s/linreg1_result_predict.csv'%self.FOUT
        pred_R, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'])

        # Test estimates
        params = lm.params[['estimate', 'stderr', 'tvalue', 'Pr(>|t|)']]
        ck = np.allclose(params, estimate)
        self.assertTrue(ck)

        # Test predictions
        y0, pint = lm.predict(pred_R[['x1', 'x2']])
        pred = pd.DataFrame({'fit':y0, 'lwr':pint['predint_025'],
                        'upr':pint['predint_975']}) 
        ck = np.allclose(pred, pred_R[['fit', 'lwr', 'upr']])
        self.assertTrue(ck)
 
    def test_ols_R2(self):
        # data set from R
        fd = '%s/linreg2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)
        fd = '%s/linreg2_result_estimate.csv'%self.FOUT
        estimate, comment = csv.read_csv(fd)
        fd = '%s/linreg2_result_predict.csv'%self.FOUT
        pred_R, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1']], data['y'], polyorder=3)

        # Test estimates
        params = lm.params[['estimate', 'stderr', 'tvalue', 'Pr(>|t|)']]
        ck = np.allclose(params, estimate)
        self.assertTrue(ck)

        # Test predictions
        y0, pint = lm.predict(pred_R[['x1']])
        pred = pd.DataFrame({'fit':y0, 'lwr':pint['predint_025'],
                        'upr':pint['predint_975']}) 
        ck = np.allclose(pred, pred_R[['fit', 'lwr', 'upr']])
        self.assertTrue(ck)

    def test_gls_ar1(self):
        # Build inputs
        nval = 50
        nvar = 3
        mu = np.random.uniform(-1,1, size=(nval, ))
        sig = 4.
        x = np.empty((nval, nvar))
        for i in range(nvar):
            x[:,i] = np.random.normal(loc=mu, scale=sig)
        rho = 0.99
        e = sutils.ar1random([rho, 8*sig*math.sqrt(1-rho**2), 0.], nval)
        theta = np.random.uniform(2, 3, size=(nvar, 1))
        y = np.dot(x, theta) + e.reshape((nval, 1))

        # Fit model
        lm1 = linreg.Linreg(x, y, type='ols', intercept=False)
        lm2 = linreg.Linreg(x, y, type='gls_ar1', intercept=False)
        # TODO : finish this test

if __name__ == "__main__":
    unittest.main()
