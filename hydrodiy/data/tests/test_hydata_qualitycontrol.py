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
        
        status = datacheck.lindetect(data)
        expected = np.array([False] * data.shape[0])
        expected[6:8] = True
        
        self.assertTrue(np.allclose(status, expected)) 


    def test_lindetect2(self):
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = 0.

        status = datacheck.lindetect(data)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected)) 


    def test_lindetect3(self):

        data = np.array([1., 2., 3., 3., 4.])
        
        status = datacheck.lindetect(data)
        expected = np.array([False] * data.shape[0])
        expected[1] = True

        self.assertTrue(np.allclose(status, expected)) 


if __name__ == "__main__":
    unittest.main()
