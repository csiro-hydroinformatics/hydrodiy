import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod import gr2m

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
        gr = gr2m.GR2M()
        gr.setoutputs(len(inputs), 9)
        gr.setparams(params)
        gr.setstates()
        gr.run(inputs)

        out = gr.getoutputs()

        cols = ['Q[mm/m]', 'ECH[mm/m]', 
           'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]',
           'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]']

        ck = np.all(out.columns.values.astype(str) == np.array(cols))
        self.assertTrue(ck)
 


if __name__ == "__main__":
    unittest.main()
