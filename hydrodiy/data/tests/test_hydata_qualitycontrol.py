import os
import unittest
import numpy as np
import pandas as pd
from hydrodiy.data import qualitycontrol

np.random.seed(0)

class QualityControlTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> QualityControlTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


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


    def test_islinear5(self):
        fd = os.path.join(self.ftest, 'islinear_test.csv')
        data = pd.read_csv(fd, comment='#', \
                    index_col=0, parse_dates=True)
        value = data.iloc[:, 0].values
        status = qualitycontrol.islinear(value, 1, eps=0.05)

        self.assertTrue(np.allclose(status, data['status']))


if __name__ == "__main__":
    unittest.main()
