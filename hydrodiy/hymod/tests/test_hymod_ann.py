import os
import re
import unittest

from timeit import Timer
import itertools
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.models.ann import ANN, CalibrationANN
from hymod.models.ann import destandardize, standardize
from hymod import calibration

# Get test data
url_testdata = 'https://drive.google.com/file/d/0B9m81HeozSRzcmNkVmdibEpmMTg'
FOUT = os.path.dirname(os.path.abspath(__file__))
ftar = '%s/rrtests.tar.gz' % FOUT
FRR = re.sub('\\.tar\\.gz', '', ftar)

if not os.path.exists(FRR):
    os.mkdir(FRR)
    req = requests.get(url_testdata, params={'alt':'media'})
    tar = tarfile.open(fileobj=req, mode='r:gz')
    tar.extractall()



class ANNTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ANNTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST


    def test_print(self):

        ann = ANN(2, 2)
        str_ann = '%s' % ann


    def test_standardize(self):

        nval = 10000
        nvar = 5
        for i in range(10):
            means = np.random.uniform(-10, 10, nvar)
            cov = np.eye(nvar)
            cov.flat[::(nvar+1)] = np.random.uniform(0, 10, nvar)

            X = np.random.multivariate_normal(means, cov, size=nval)
            c = 1e-5
            X = np.exp(X)-c

            Un, mu, su = standardize(X, c)
            X2 = destandardize(Un, mu, su, c)

            ck = np.allclose(X, X2)
            self.assertTrue(ck)


    def test_run1(self):

        n1 = [10, 100, 10000]
        n2 = [1, 2, 50]
        n3 = [1, 2, 10]

        for nval, ninputs, nneurons in itertools.product(n1, n2, n3):

            inputs = np.random.uniform(-1, 1, (nval, ninputs))

            # Parameters
            ann = ANN(ninputs, nneurons)

            params = np.random.uniform(-1, 1, ann.params.nval)
            ann.params.data = params
            ann.allocate(len(inputs), 2)
            ann.inputs.data = inputs
            ann.run()

            Q1 = ann.outputs.data[:, 0]
            
            L1M, L1C, L2M, L2C = ann.params2matrix()
            S2 = np.tanh(np.dot(inputs, L1M) + L1C)
            Q2 = (np.dot(S2, L2M) + L2C)[:,0]

            self.assertTrue(np.allclose(Q1, Q2))


    def test_calibrate(self):

        ninputs = 2
        nneurons = 3
        calib = CalibrationANN(ninputs, nneurons)
        calib.errfun = calibration.ssqe_bias

        for count in range(1, 11):
            fd = '%s/rrtest_%2.2d_timeseries.csv' % (FRR, count)
            d, comment = csv.read_csv(fd, index_col=0, \
                    parse_dates=True)
            idx = d['obs']>=0
            d = d[idx]
            dm = d.resample('MS', how='sum')

            inputs = dm[['rainfall', 'obs']].values
            obs = dm['obs'].shift(-1).values

            # Standardize
            cst = 0.01
            inputs_s, m_I, s_I = standardize(inputs, cst)
            obs_s, m_O, s_O = standardize(obs, cst)

            # Run gr first
            calib.setup(obs_s, inputs_s)
            calib.idx_cal = pd.notnull(obs_s)

            ann = calib.model
            params = np.random.uniform(-1, 1, ann.params.nval)
            ann.params.data = params
            ann.run()
            calib.observations.data = ann.outputs.data[:,0].copy()

            # Calibrate
            ini, explo, explo_ofun = calib.explore()
            final, _, _ = calib.fit(ini)

            err = np.abs(calib.model.params.data - params)
            ck = np.max(err[[0, 2]]) < 1
            ck = ck & (err[1] < 1e-1)
            ck = ck & (err[3] < 1e-2)

            print('\t\tTEST CALIB %2d : max abs err = %0.5f' % ( \
                    count, np.max(err)))

            self.assertTrue(ck)





if __name__ == "__main__":
    unittest.main()

