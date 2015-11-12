import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.models.turcmezentsev import TurcMezentsev, CalibrationTurcMezentsev

class TurcMezentsevTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> TurcMezentsevTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_print(self):
        tm = TurcMezentsev()
        str_tm = '%s' % tm


    def test_run(self):
        P = np.linspace(700, 1200, 20)
        PE = np.linspace(1000, 2000, 20)
        inputs = np.concatenate([P[:,None], PE[:, None]], axis=1)

        tm = TurcMezentsev()
        tm.allocate(len(inputs), 2)
        tm.inputs.data = inputs
        tm.run()
        Q1 = tm.outputs.data[:, 0]
        n = tm.params['n']
        Q2 = P*(1-1/(1+(P/PE)**n)**(1/n))
        self.assertTrue(np.allclose(Q1, Q2))


    def test_calibrate(self):
        Q = [85.5, \
                331.7, \
                87.5, \
                109.6, \
                60.3, \
                122.4, \
                87.9, \
                57.8, \
                290.2, \
                304.4, \
                24.8, \
                503.4, \
                261.4, \
                626.4, \
                206.4, \
                165.6, \
                348.5, \
                329.5, \
                214.9, \
                297.4, \
                507.2, \
                164.4]

        P = [749.2, \
                1142.0, \
                896.3, \
                820.8, \
                735.7, \
                901.1, \
                800.6, \
                808.5, \
                956.9, \
                954.2, \
                668.7, \
                1152.4, \
                1049.3, \
                1324.2, \
                884.7, \
                783.9, \
                927.0, \
                1034.7, \
                1124.2, \
                841.5, \
                922.4, \
                1020.1]

        E = [1173.7, \
                1110.8, \
                1150.5, \
                1120.3, \
                1141.2, \
                1362.1, \
                1237.0, \
                1217.7, \
                1172.7, \
                1253.1, \
                1421.2, \
                1246.0, \
                1333.7, \
                1116.4, \
                1167.3, \
                1135.9, \
                1177.4, \
                1171.0, \
                1231.1, \
                1068.3, \
                1006.3, \
                1171.0]

        inputs = np.array([P, E]).T
        obs = np.array(Q)
        calib = CalibrationTurcMezentsev()
        calib.setup(obs, inputs)
        calib.idx_cal = obs == obs

        start, _, _ = calib.explore()
        end, _, _ = calib.fit(start)

        tm = calib.model

        ck = np.allclose(tm.params['n'], 1.753469, atol=1e-3)
        if not ck:
            print('tm.trueparams = {0}'.format(tm.params['n']))

        self.assertTrue(ck)




if __name__ == "__main__":
    unittest.main()

