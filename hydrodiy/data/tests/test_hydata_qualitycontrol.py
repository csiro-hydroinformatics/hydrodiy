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


    def test_islinear_linspace(self):
        ''' Test is linear against the numpy linspace function '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)

        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])
        expected[4:16] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_linspace_npoints(self):
        ''' Test is linear against the numpy linspace function '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)

        for npoints in range(2, 5):
            status = qualitycontrol.islinear(data, npoints)

            expected = np.array([False] * data.shape[0])
            expected[3+npoints:17-npoints] = True

            ck = np.allclose(status, expected)
            self.assertTrue(ck)


    def test_islinear_zeros(self):
        ''' Test if islinear is sensitive to zeros '''
        nval = 20
        data = np.random.normal(size=nval)
        data[5:9] = 0.

        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_int(self):
        ''' Test is islinear against integer '''
        data = np.array([1., 2., 3., 3., 4., 3.])
        status = qualitycontrol.islinear(data)
        expected = np.array([False] * data.shape[0])
        expected[1] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_int_2points(self):
        ''' Test two points interpolation against integer '''
        data = np.array([1., 2., 3., 3., 4., 3.])
        status = qualitycontrol.islinear(data, 2)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_casestudy(self):
        ''' Test known dataset '''
        fd = os.path.join(self.ftest, 'islinear_test.csv')
        data = pd.read_csv(fd, comment='#', \
                    index_col=0, parse_dates=True)
        value = data.iloc[:, 0].values
        status = qualitycontrol.islinear(value, 1, tol=0.05)

        self.assertTrue(np.allclose(status, data['status']))


if __name__ == "__main__":
    unittest.main()
