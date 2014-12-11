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
        lm.fit()

        print(lm)

    def test_ols_johnston(self):
        # data set from Johnston and Di Nardo, page 75
        # Econometrics Methods, 1993
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit()

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
        lm.fit()

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
        lm.fit()

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

    def test_gls_johnston(self):
        # data set from Johnston and Di Nardo, page 193
        # Econometrics Methods, 1993

        # Build inputs
        nval = 40

        x = 10. + 5.*np.random.normal(size=(nval+1,1))
        x[0] = 5.
        XX = np.concatenate([x[1:], x[:-1]], axis=1)

        e = 2 + 2.*XX[:,0] - 0.5*XX[:,1] 
        y = sutils.ar1innov([0.7, 0.], e) + 5*np.random.normal(size=(nval,))
        ym = np.concatenate([np.array([0.]), y[:-1]]).reshape((nval, 1))
        XX = np.concatenate([XX, ym], axis=1)

        # Fit model
        #lm1 = linreg.Linreg(XX, y, type='ols')  # Correct regression
        lm2 = linreg.Linreg(XX[:,0], y, type='ols') # Misspecified
        lm2.fit()
        y2, int2 = lm2.predict(XX[:,0])

        lm3 = linreg.Linreg(XX[:,0], y, type='gls_ar1') # GLS temptative
        lm3.fit()
        y3, int3 = lm3.predict(XX[:,0])

        import matplotlib.pyplot as plt
        plt.plot(y, label = 'obs')
        plt.plot(y2, label = 'lm2')
        plt.plot(y3, label = 'lm3')
        plt.legend()

        import pdb; pdb.set_trace()

        # TODO : finish this test

if __name__ == "__main__":
    unittest.main()
