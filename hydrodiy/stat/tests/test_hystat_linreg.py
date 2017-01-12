import os
import math
import unittest
import logging

import warnings
warnings.filterwarnings('error')

import numpy as np
import pandas as pd

from hydrodiy.io import csv
from hydrodiy.stat import linreg
from hydrodiy.stat import sutils

# set numpy seed
np.random.seed(100)

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
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

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
        fd = '%s/data/olslinreg1_data.csv'%self.ftest
        data, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg1_result_estimate.csv' % self.ftest
        estimate, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg1_result_predict.csv' % self.ftest
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
        fd = '%s/data/olslinreg2_data.csv'%self.ftest
        data, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg2_result_estimate.csv' % self.ftest
        estimate, comment = csv.read_csv(fd)

        fd = '%s/data/olslinreg2_result_predict.csv' % self.ftest
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


    def test_gls_rcran(self):
        # data set from R - see linreg_gls.r
        for itest in range(1, 3):
            fd = '%s/data/glslinreg%d_data.csv' % (self.ftest, itest)
            data, comment = csv.read_csv(fd)

            fd = ('%s/data/glslinreg%d_result_estimate'
                    '_gls.csv') % (self.ftest, itest)
            estimate, comment = csv.read_csv(fd)

            fd = ('%s/data/glslinreg%d_result_predict_'
                    'gls.csv') % (self.ftest, itest)
            pred_R, comment = csv.read_csv(fd)

            # Fit model
            lm = linreg.Linreg(data[['x1', 'x2']], data['y'],
                    regtype='gls_ar1')
            lm.fit(False)

            # Test estimates
            params = lm.params['estimate'].values
            expected = estimate['Estimate'].values

            ck = np.allclose(params, expected, atol=6e-2)
            self.assertTrue(ck)

            # Correct for intercept
            # (not well determined in gls ar1 regressions)
            lm.params.loc['intercept', 'estimate'] = expected[0]

            # Test predictions
            y0, pint = lm.predict(pred_R[['x1', 'x2']])
            ck = np.allclose(y0, pred_R['gls'], atol=5e-1)
            self.assertTrue(ck)


    def test_gls_elasticity(self):
        fd = '%s/data/elasticity_data.csv' % self.ftest
        data, comment = csv.read_csv(fd)

        # Fit model
        lm = linreg.Linreg(data[['x1', 'x2']], data['y'],
                regtype='gls_ar1',
                has_intercept=False)

        # Initial tests were producing an error
        lm.fit()


    def test_boot_ols(self):
        ''' Test bootstrap on OLS regression '''
        # data set from R - see linreg.r
        #fd = '%s/data/olslinreg1_data.csv'%self.ftest
        #data, comment = csv.read_csv(fd)

        # Simple normal data
        nval = 100
        x1 = 1+2*np.random.normal(size=nval)
        x2 = 2+1.5*np.random.normal(size=nval)
        y = 5+4*x1+5*x2+0.1*np.random.normal(size=nval)
        data = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})

        # Fit model
        lm1 = linreg.Linreg(data[['x1', 'x2']], data['y'])
        lm1.fit()

        lm2 = linreg.Linreg(data[['x1', 'x2']], data['y'])
        lm2.boot(nsample=500)

        # parameters
        for nm in ['estimate', '2.5%', '97.5%']:
            p1 = lm1.params[nm]
            if nm == 'estimate':
                p2 = lm2.params_boot.median()
            else:
                p2 = lm2.params_boot.apply(np.percentile,
                                args=(float(nm[:-1]),))
            ck1 = np.allclose(p1, p2, atol=5e-2)
            self.assertTrue(ck1)

        # Predictions
        y1, ci1 = lm1.predict(boot=False)
        y2, ci2 = lm2.predict(boot=True)
        ck2 = np.allclose(y1, y2, atol=5e-2)
        ck3 = np.allclose(ci1, ci2, atol=3e-1)

        self.assertTrue(ck2)
        self.assertTrue(ck3)


    def test_print_boot_ols(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit(False)

        lm.boot(nsample=500)
        print(lm)


    def test_leverages(self):
        x = [[3, 5], [1, 4], [5, 6], [2, 4], [4, 6]]
        y = [3, 1, 8, 3, 5]

        lm = linreg.Linreg(x, y)
        lm.fit()
        lev, cook, std_res = lm.leverages()

        xx = np.insert(np.array(x), 0, 1., axis=1)
        tXXinv = np.linalg.inv(np.dot(xx.T, xx))
        lev_expected = np.diag(np.dot(xx, np.dot(tXXinv, xx.T)))

        ck = np.allclose(lev, lev_expected)
        self.assertTrue(ck)


    def test_big(self):
        ''' Test a regression on a large dataset '''
        nval = 100000
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


