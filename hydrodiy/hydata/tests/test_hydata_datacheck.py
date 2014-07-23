import os
import unittest
import numpy as np
import pandas as pd
from hydata import datacheck

class DataCheckTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> DataCheckTestCase')
        FTEST, testfile = os.path.split(__file__)

    def test_lindetect1(self):
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = np.linspace(0, 1, 4)
        linstatus = datacheck.lindetect(data, params=[1, 1e-5])
        expected = np.zeros(data.shape[0],int)
        expected[6:8] = 1
        self.assertTrue(np.allclose(linstatus, expected)) 

    def test_lindetect2(self):
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = 0.
        linstatus = datacheck.lindetect(data, params=[1, 1e-5])
        expected = np.zeros(data.shape[0],int)
        self.assertTrue(np.allclose(linstatus, expected)) 

if __name__ == "__main__":
    unittest.main()
