import os
import unittest
import numpy as np
import pandas as pd
from hydrodiy.data import qualitycontrol as qc

np.random.seed(0)

class QualityControlTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> QualityControlTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_islinear_1d_linspace(self):
        ''' Test is linear 1d against the numpy linspace function '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)

        status = qc.islinear(data, npoints=1)
        expected = np.array([False] * data.shape[0])
        expected[4:16] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_linspace_npoints(self):
        ''' Test is linear 1d against the numpy linspace function '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)

        for npoints in range(2, 5):
            status = qc.islinear(data, npoints)

            expected = np.array([False] * data.shape[0])
            expected[3:16] = True

            ck = np.allclose(status, expected)
            if not ck:
                import pdb; pdb.set_trace()
            self.assertTrue(ck)


    def test_islinear_1d_zeros(self):
        ''' Test if islinear 1d is sensitive to zeros '''
        nval = 20

        data = np.random.normal(size=nval)
        data[5:9] = 0.
        status = qc.islinear(data, npoints=1)
        expected = np.array([False] * data.shape[0])
        expected[5:9] = True
        self.assertTrue(np.allclose(status, expected))

        status = qc.islinear(data, npoints=1, thresh=0.)
        expected = np.array([False] * data.shape[0])
        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_int(self):
        ''' Test is islinear 1d against integer '''
        data = np.array([1., 2., 3., 3., 4., 3.])
        status = qc.islinear(data, npoints=1)
        expected = np.array([False] * data.shape[0])
        expected[1] = True

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_int_2points(self):
        ''' Test two points interpolation against integer '''
        data = np.array([1., 2., 3., 3., 4., 3.])
        status = qc.islinear(data, npoints=2)
        expected = np.array([False] * data.shape[0])

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_casestudy(self):
        ''' Test known dataset '''
        fd = os.path.join(self.ftest, 'islinear_test.csv')
        data = pd.read_csv(fd, comment='#', \
                    index_col=0, parse_dates=True)
        value = data.iloc[:, 0].values
        status = qc.islinear(value, 1, tol=0.05)

        self.assertTrue(np.allclose(status, data['status']))


if __name__ == "__main__":
    unittest.main()
