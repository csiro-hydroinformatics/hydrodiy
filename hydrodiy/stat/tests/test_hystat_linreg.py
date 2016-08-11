import os
import math
import unittest
import logging

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.io import csv
from hydrodiy.stat import linreg
from hydrodiy.stat import sutils

show_log = True

if show_log:
    LOGGER = logging.getLogger('hydrodiy.stat.linreg')

    LOGGER.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    ft = logging.Formatter('%(asctime)s - %(message)s')
    sh.setFormatter(ft)


class LinregTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> LinregTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        self.y1 = np.array(1)

    def test_aprint(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]
        lm = linreg.Linreg(x, y)
        lm.fit()


    def test_ols_johnston(self):
        # data set from Johnston and Di Nardo, page 75
        # Econometrics Methods, 1993
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit(False)

        self.assertTrue(np.allclose(lm.params['estimate'].values,
                                np.array([4., 2.5, -1.5])))

        sigma = lm.sigma
        self.assertTrue(sigma==math.sqrt(0.75))

        ck = np.allclose(lm.params['stderr'][2], sigma*math.sqrt(2.5))
        self.assertTrue(ck)

        ci = lm.params[['2.5%', '97.5%']][-1:]
        ck = np.allclose(ci, [-7.39, 4.39], atol=1e-2)
        self.assertTrue(ck)

        y0, pint = lm.predict(np.array([10, 10]).reshape((1,2)))
        self.assertTrue(np.allclose(y0[0],14))


    def test_ols_rcran1(self):

        # data set from R - see linreg.r
        fd = '%s/data/olslinreg1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg1_result_estimate.csv' % self.FOUT
        estimate, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg1_result_predict.csv' % self.FOUT
        pred_R, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'])
        lm.fit(False)

        # Test estimates
        params = lm.params[['estimate', 'stderr', 'tvalue', 'Pr(>|t|)']]
        ck = np.allclose(params, estimate)
        self.assertTrue(ck)

        # Test predictions
        y0, pint = lm.predict(pred_R[['x1', 'x2']])
        pred = pd.DataFrame({'fit':y0, 'lwr':pint['2.5%'],
                        'upr':pint['97.5%']})
        ck = np.allclose(pred, pred_R[['fit', 'lwr', 'upr']])
        self.assertTrue(ck)


    def test_ols_rcran2(self):

        # data set from R - see linreg.r
        fd = '%s/data/olslinreg2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg2_result_estimate.csv' % self.FOUT
        estimate, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg2_result_predict.csv' % self.FOUT
        pred_R, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data['x1'], data['y'], polyorder=3)
        lm.fit(False)

        # Test estimates
        params = lm.params[['estimate', 'stderr', 'tvalue', 'Pr(>|t|)']]
        ck = np.allclose(params, estimate)
        self.assertTrue(ck)

        # Test predictions
        y0, pint = lm.predict(pred_R['x1'])
        pred = pd.DataFrame({'fit':y0, 'lwr':pint['2.5%'],
                        'upr':pint['97.5%']})
        ck = np.allclose(pred, pred_R[['fit', 'lwr', 'upr']])
        self.assertTrue(ck)


    def test_ols_scatterplot(self):

        # data set from R - see linreg.r
        fd = '%s/data/olslinreg2_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data['x1'], data['y'], polyorder=3)
        lm.fit(log_entry=False)

        # Plot
        fig, ax = plt.subplots()
        lm.scatterplot(ax=ax)
        fp = '%s/data/olslinreg2_scatter.png'%self.FOUT
        fig.savefig(fp)

        # Plot
        lm.boot()
        fig, ax = plt.subplots()
        lm.scatterplot(ax=ax, boot=True)
        fp = '%s/data/olslinreg2_scatter_boot.png'%self.FOUT


    def test_gls_rcran(self):

        # data set from R - see linreg_gls.r
        for itest in range(1, 3):

            fd = '%s/data/glslinreg%d_data.csv' % (self.FOUT, itest)
            data, comment = csv.read_csv(fd)

            fd = ('%s/data/glslinreg%d_result_estimate'
                    '_gls.csv') % (self.FOUT, itest)
            estimate, comment = csv.read_csv(fd)

            fd = ('%s/data/glslinreg%d_result_predict_'
                    'gls.csv') % (self.FOUT, itest)
            pred_R, comment = csv.read_csv(fd)

            # Fit model
            lm = linreg.Linreg(data[['x1', 'x2']], data['y'],
                    regtype='gls_ar1')
            lm.fit(False)

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
        ''' Test bootstrap on OLS regression '''

        # data set from R - see linreg.r
        fd = '%s/data/olslinreg1_data.csv'%self.FOUT
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'])
        lm.boot(nsample=500)

        p1 = lm.params['estimate']
        p2 = lm.params_boot.median()
        ck1 = np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck1)

        p1 = lm.params['2.5%']
        p2 = lm.params_boot.apply(lambda x:
                    sutils.percentiles(x, 2.5))
        ck2= np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck2)

        p1 = lm.params['97.5%']
        p2 = lm.params_boot.apply(lambda x:
                    sutils.percentiles(x, 97.5))
        ck3 = np.allclose(p1, p2, atol=2e-1)
        self.assertTrue(ck3)


    def test_boot_gls(self):
        ''' Test bootstrap on GLS regression '''

        itest = 4
        fd = '%s/data/glslinreg%d_data.csv' % (self.FOUT, itest)
        data, comment = csv.read_csv(fd)

        # Fit model
        cc = [cn for cn in data.columns if cn != 'y']
        lm = linreg.Linreg(data[cc], data['y'], regtype='gls_ar1')
        lm.boot(nsample=200)


    def test_print_boot_ols(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit(False)

        lm.boot(nsample=1000)
        print(lm)


    def test_leverages(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lev = lm.leverages()

        xx = np.array(xx)
        tXXinv = np.linalg.inv(np.dot(x.T, x))
        lev_expected = np.diag(np.dot(x, np.dot(tXXinv, x.T)))

        self.assertTrue(np.allclose(lev, lev_expected))


    def test_predict_boot_ols(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit(False)

        lm.boot(nsample=1000)

        yhat1, pred1 = lm.predict()
        yhat2, pred2 = lm.predict(boot=True)

        # TODO !!!


    def test_big(self):
        ''' Test a regression on a large dataset '''
        nval = 1000000
        npred = 50
        x = np.random.normal(0, 2, size=(nval, npred))
        theta = np.random.uniform(-1, 1, npred+1)

        y0 = theta[0] + np.dot(x, theta[1:].reshape(npred,1))
        eps = np.random.normal(0, np.std(y0)/10, len(y0))
        y = y0 + eps.reshape((nval, 1))

        lm = linreg.Linreg(x, y)
        lm.fit(False)
        yhat = lm.predict()


if __name__ == "__main__":
    unittest.main()


