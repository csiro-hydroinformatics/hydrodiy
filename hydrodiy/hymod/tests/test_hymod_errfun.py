import os
import re
import unittest

import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hyio import csv
from hymod import calibration


class ErrfunTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> ErrfunTestCase')


    def test_sse(self):
        nval = 1000
        obs = np.random.uniform(size=nval)
        sim = np.random.uniform(size=nval)
        err1 = calibration.sse(obs, sim, None)
        err2 = np.sum((obs-sim)**2)
        ck = abs(err1-err2) < 1e-10
        self.assertTrue(ck)


    def test_ssqe_bias(self):
        nval = 1000
        obs = np.random.uniform(size=nval)
        sim = np.random.uniform(size=nval)
        err1 = calibration.ssqe_bias(obs, sim, None)
        err2 = np.sum((np.sqrt(obs)-np.sqrt(sim))**2)
        err2 *= (1+abs(np.mean(obs)-np.mean(sim)))
        ck = abs(err1-err2) < 1e-10
        self.assertTrue(ck)




if __name__ == "__main__":
    unittest.main()
