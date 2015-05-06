import os
import unittest
import numpy as np
import pandas as pd
from hyio import binfile

class BinfileTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BinTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_binfile1(self):

        nrow = 100
        ncol = 5
        data1 = pd.DataFrame(np.random.normal(size=(nrow, ncol)))

        ft = '%s/binfile_testdata.bin'%self.FOUT        
        binfile.write_bin(data1, ft, 'test data')

        data2, comment = binfile.read_bin(ft)

        self.assertTrue(np.allclose(data1, data2))
        self.assertTrue(start==start2)
        self.assertTrue(dt==dt2)

if __name__ == "__main__":
    unittest.main()
