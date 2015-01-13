import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
 
    def test_ols_rcran1(self):

        # data set from R - see linreg.r
        fd = '%s/olslinreg1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        fd = '%s/olslinreg1_result_estimate.csv' % self.FOUT
        estimate, comment = csv.read_csv(fd)

        fd = '%s/olslinreg1_result_predict.csv' % self.FOUT
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
 
    def test_ols_rcran2(self):

        # data set from R - see linreg.r
        fd = '%s/olslinreg2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        fd = '%s/olslinreg2_result_estimate.csv' % self.FOUT
        estimate, comment = csv.read_csv(fd)

        fd = '%s/olslinreg2_result_predict.csv' % self.FOUT
        pred_R, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data['x1'], data['y'], polyorder=3)
        lm.fit()

        # Test estimates
        params = lm.params[['estimate', 'stderr', 'tvalue', 'Pr(>|t|)']]
        ck = np.allclose(params, estimate)
        self.assertTrue(ck)

        # Test predictions
        y0, pint = lm.predict(pred_R['x1'])
        pred = pd.DataFrame({'fit':y0, 'lwr':pint['predint_025'],
                        'upr':pint['predint_975']}) 
        ck = np.allclose(pred, pred_R[['fit', 'lwr', 'upr']])
        self.assertTrue(ck)
  
    def test_ols_scatterplot(self):

        # data set from R - see linreg.r
        fd = '%s/olslinreg2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data['x1'], data['y'], polyorder=3)
        lm.fit()

        # Plot
        fig, ax = plt.subplots()
        lm.scatterplot(ax)
        fp = '%s/olslinreg2_scatter_normal.png'%self.FOUT
        fig.savefig(fp)

        lm.scatterplot(ax, log=True)
        fp = '%s/olslinreg2_scatter_log.png'%self.FOUT
        fig.savefig(fp)


    def test_gls_rcran(self):

        # data set from R - see linreg_gls.r
        for itest in range(1, 3):

            fd = '%s/glslinreg%d_data.csv' % (self.FOUT, itest)
            data, comment = csv.read_csv(fd)

            fd = '%s/glslinreg%d_result_estimate_gls.csv' % (self.FOUT, itest)
            estimate, comment = csv.read_csv(fd)

            fd = '%s/glslinreg%d_result_predict_gls.csv' % (self.FOUT, itest)
            pred_R, comment = csv.read_csv(fd)

            # Fit model
            lm = linreg.Linreg(data[['x1', 'x2']], data['y'], type='gls_ar1')
            lm.fit()

            # Test estimates
            params = lm.params['estimate']
            estimate = estimate['Estimate']

            ck1 = np.allclose(params[1:], estimate[1:], atol=6e-2)
            self.assertTrue(ck1)

            ck2 = np.allclose(params[0], estimate[0], atol=2e-1)
            if itest==1:
                self.assertTrue(ck2)

            # Test predictions
            y0, pint = lm.predict(pred_R[['x1', 'x2']])
            check = np.abs(y0-pred_R['gls'])/(1+np.abs(y0))
            idx = [i!=5 for i in range(len(check))]
            #ck = np.all(check[idx]<0.1)
            #if itest==2:
            #    self.assertTrue(ck)
 
    def test_boot_ols(self):

        # data set from R - see linreg.r
        fd = '%s/olslinreg1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'])
        lm.boot(nsample=500)

        p1 = lm.params['estimate']
        p2 = lm.params_boot_percentiles['P_50.0']
        ck1 = np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck1)

        p1 = lm.params['confint_025']
        p2 = lm.params_boot_percentiles['P__2.5']
        ck2= np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck2)

        p1 = lm.params['confint_975']
        p2 = lm.params_boot_percentiles['P_97.5']
        ck3 = np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck3)
     
    def test_boot_gls(self):

        # data set from R - see linreg_gls.r
        itest = 3
        fd = '%s/glslinreg%d_data.csv' % (self.FOUT, itest)
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'], type='gls_ar1')
        lm.boot(nsample=100)

if __name__ == "__main__":
    unittest.main()


