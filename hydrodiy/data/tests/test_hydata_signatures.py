import os
import time
import unittest
import numpy as np
import pandas as pd

from hydrodiy.data import signatures
from hydrodiy.io import csv

class SignaturesTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SignaturesTestCase (hydata)')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_eckhardt(self):
        ''' Test Eckhardt baseflow '''

        fd = os.path.join(self.ftest, 'baseflow_RDF_EC.csv')
        data, _  = csv.read_csv(fd)
        flow = data.iloc[:, 2]
        bflow_expected = data.iloc[:, 3]

        bflow = signatures.eckhardt(flow, \
                        tau=100,\
                        thresh=0.95, \
                        BFI_max = 0.80)

        self.assertTrue(np.allclose(bflow_expected, bflow))


    def test_fdcslope(self):
        ''' Test fdc slope computation '''
        x = np.linspace(0, 1, 101)
        slp, qq = signatures.fdcslope(x, q1=90, q2=100, cst=0.5)
        self.assertTrue(np.isclose(slp, 0.99009901))
        self.assertTrue(np.allclose(qq, [0.9, 1]))


    def test_fdcslope_error(self):
        ''' Test fdc slope computation error '''
        x = np.linspace(0, 1, 101)
        try:
            slp = signatures.fdcslope(x, q1=90, q2=80, cst=0.5)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected q2 > q1'))
        else:
            raise ValueError('Problem with error handling')


    def test_goue(self):
        ''' Test goue computation '''
        dt = pd.date_range('2000-01-10', '2000-03-31')
        nt = len(dt)
        daily = pd.Series(np.random.uniform(0, 1, nt), index=dt)

        gv, dflat, m = signatures.goue(daily)



if __name__ == "__main__":
    unittest.main()
