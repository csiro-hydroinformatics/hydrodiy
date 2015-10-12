import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.gr2m import GR2M

class GR2MTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR2MTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

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
        gr.set_states()
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
        gr.set_states()
        gr.run(inputs)
        out = gr.get_outputs()
        res = out.values[12:,]

        # Test
        expected = data.loc[:, ['DebitSimule', 'F', 'P1', \
                'P2', 'P3', 'R1', 'R2', 'S', 'R']]
        expected = expected.values[12:,:]
        ck = np.allclose(res, expected)
        self.assertTrue(ck)
 


if __name__ == "__main__":
    unittest.main()
