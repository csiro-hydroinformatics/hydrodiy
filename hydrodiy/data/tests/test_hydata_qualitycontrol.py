import os
import unittest
import numpy as np
import pandas as pd
from hydrodiy.data import qualitycontrol

class DataCheckTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> DataCheckTestCase')
        FTEST, testfile = os.path.split(__file__)


    def test_islinear1(self):
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = np.linspace(0, 1, 4)

        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])
        expected[6:8] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear2(self):
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = 0.

        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected))


    def test_islinear3(self):

        data = np.array([1., 2., 3., 3., 4., 3.])

        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])
        expected[1] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear4(self):

        data = np.array([1., 2., 3., 3., 4., 3.])

        status = qualitycontrol.islinear(data, 2)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected))



if __name__ == "__main__":
    unittest.main()
