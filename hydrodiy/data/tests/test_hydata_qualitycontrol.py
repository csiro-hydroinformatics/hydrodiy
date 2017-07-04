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


    def test_islinear_error(self):
        ''' Test is linear errors '''
        nval = 20
        data = np.random.normal(size=nval)

        try:
            status = qc.islinear(data, npoints=0)
        except Exception as err:
            pass
        self.assertTrue(str(err).startswith('Expected npoints'))

        try:
            status = qc.islinear(data, tol=1e-11)
        except Exception as err:
            pass
        self.assertTrue(str(err).startswith('Expected tol'))


    def test_islinear_1d_linspace(self):
        ''' Test is linear 1d against the numpy linspace function '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)

        status = qc.islinear(data, npoints=1)
        expected = np.zeros(data.shape[0])
        expected[3:17] = 1

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_constant(self):
        ''' Test is linear 1d against constant data '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = 100

        status = qc.islinear(data, npoints=1)
        expected = np.zeros(data.shape[0])
        expected[3:17] = 2

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_nan(self):
        ''' Test is linear 1d against nan data '''
        nval = 20
        data = np.random.normal(size=nval)
        data[3:17] = np.linspace(0, 1, 14)
        data[12:16] = np.nan

        status = qc.islinear(data, npoints=1)
        expected = np.zeros(data.shape[0])
        expected[3:12] = 1

        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_linspace_npoints(self):
        ''' Test is linear 1d against the numpy linspace function '''
        nval = 30
        data = np.random.normal(size=nval)

        i1 = 10
        i2 = 20
        idxlin = np.arange(i1, i2+1)
        data[idxlin] = np.linspace(0, 1, 14)

        for npoints in range(2, 5):
            status = qc.islinear(data, npoints)

            expected = np.zeros(data.shape[0])
            expected[i1:i2+1] = 1

            ck = np.allclose(status, expected)
            self.assertTrue(ck)


    def test_islinear_1d_zeros(self):
        ''' Test if islinear 1d is sensitive to zeros '''
        nval = 20

        data = np.random.normal(size=nval)
        data[5:9] = 0.
        status = qc.islinear(data, npoints=1, thresh=data.min()-1)
        expected = np.zeros(data.shape[0])
        expected[5:9] = 2
        self.assertTrue(np.allclose(status, expected))

        status = qc.islinear(data, npoints=1, thresh=0.)
        expected = np.array([False] * data.shape[0])
        self.assertTrue(np.allclose(status, expected))


    def test_islinear_1d_int(self):
        ''' Test is islinear 1d against integer '''
        data = np.array([0.]*20+[0., 1., 2., 3., 4., 5., 3.]+[0.]*20)
        for npoints in range(1, 7):
            status = qc.islinear(data, npoints=npoints)
            expected = np.zeros(data.shape[0])
            if npoints<=4:
                expected[20:26] = 1

            ck = np.allclose(status, expected)
            self.assertTrue(ck)


    def test_islinear_sequence(self):
        ''' Test is_linear for two consecutive sequences of linear data '''
        nval = 50
        data = np.random.uniform(size=nval)
        ia1, ia2 = 20, 24
        data[ia1:ia2+1] = np.linspace(0, 1, 5)

        ib1, ib2 = 26, 31
        data[ib1:ib2+1] = np.linspace(0, 1, 6)

        for npoints in range(1, 10):
            status = qc.islinear(data, npoints)

            expected = np.zeros(nval)
            if npoints <= 3:
                expected[ia1:ia2+1] = 1

            if npoints <= 4:
                expected[ib1:ib2+1] = 1

            ck = np.allclose(status, expected)
            self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
