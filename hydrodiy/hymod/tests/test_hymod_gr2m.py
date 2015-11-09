import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.models.gr2m import GR2M
from hymod.model import Calibration
from hymod import errfun

class GR2MTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR2MTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_print(self):
        gr = GR2M()
        str_gr = '%s' % gr

    def test_get_calparams_sample(self):
        nsamples = 100
        gr = GR2M()
        samples = gr.get_calparams_samples(nsamples)
        self.assertTrue(samples.shape == (nsamples, 2))


    def test_gr2m_dumb(self):

        nval = 100
        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)

        params = [400, 0.9]

        # Run
        gr = GR2M()
        gr.create_outputs(len(inputs), 9)
        gr.set_trueparams(params)
        gr.initialise()
        gr.run(inputs)

        out = gr.get_outputs()

        cols = ['Q[mm/m]', 'Ech[mm/m]', 
           'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]',
           'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]']

        ck = np.all(out.columns.values.astype(str) == np.array(cols))
        self.assertTrue(ck)
 

    def test_gr2m_irstea(self):

        fd = '%s/GR2M.csv' % self.FOUT
        data, comment =csv.read_csv(fd)
        inputs = data.loc[:, ['Pluie (mm)', 'ETP (mm)']].values
        inputs = np.ascontiguousarray(inputs)

        params = [650.7, 0.8]

        # Run
        gr = GR2M()
        gr.create_outputs(len(inputs), 9)
        gr.set_trueparams(params)
        gr.initialise()
        gr.run(inputs)
        out = gr.get_outputs()

        # Test
        warmup = 30
        res = out.values[warmup:,]
        expected = data.loc[:, ['DebitSimule', 'F', 'P1', \
                'P2', 'P3', 'R1', 'R2', 'S', 'R']]
        expected = expected.values[warmup:,:]

        for i in range(res.shape[1]):
            err = np.abs(res[:,i] - expected[:,i])

            err_thresh = 4e-1
            if not i in [0, 1, 6]:
                err_thresh = 1e-2 * np.min(np.abs(expected[expected[:,i]!=0.,i]))
            ck = np.max(err) < err_thresh

            if not ck:
                print('\tVAR[%d] : max abs err = %0.5f < %0.5f ? %s' % ( \
                        i, np.max(err), err_thresh, ck)) 

            self.assertTrue(ck)
 

    def test_gr2m_irstea_calib(self):

        fd = '%s/GR2M.csv' % self.FOUT
        data, comment =csv.read_csv(fd)
        inputs = data.loc[:, ['Pluie (mm)', 'ETP (mm)']].values
        inputs = np.ascontiguousarray(inputs)

        calparams_expected = [650.7, 0.8]

        gr = GR2M()
        gr.create_outputs(len(inputs), 1)

        # Parameter samples
        nsamples = 200
        samples = gr.get_calparams_samples(nsamples)

        # loop through parameters
        for i in range(nsamples):

            # Generate obs
            gr.set_calparams(samples[i, :])
            expected = gr.trueparams.copy()
            gr.initialise()
            gr.run(inputs)
            obs = gr.outputs[:,0].copy()
            idx_cal = np.arange(12, len(inputs))

            # Calibrate
            calib = Calibration(gr, inputs, obs)
            calib.set_errfun(errfun.ssqe_bias)
            calib.set_idx_cal(idx_cal)
                        
            calparams_ini, _, _ = calib.explore()
            calparams_final, _, _ = calib.fit(calparams_ini)


            err = np.abs(gr.trueparams-expected)
            ck = np.max(err) < 1e-5

            self.assertTrue(ck)



if __name__ == "__main__":
    unittest.main()
