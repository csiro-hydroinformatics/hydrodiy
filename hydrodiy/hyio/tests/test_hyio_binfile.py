import os
import re
import unittest
import numpy as np
import pandas as pd
from hyio import binfile

class BinfileTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BinTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

        self.nrow = 100
        self.ncol = 5
        
    def test_binfile1(self):
        """ Binfile with double data """

        data1 = pd.DataFrame(np.random.normal(size=(self.nrow, self.ncol)))

        ft = '%s/binfile_testdata1.bin'%self.FOUT        
        binfile.write_bin(data1, ft, 'test data')

        data2, comment = binfile.read_bin(ft)

        self.assertTrue(np.allclose(data1, data2))
        
        F = self.FOUT
        cmd = 'rm %s/*.bind %s/*.bins %s/*.binl' % (F, F, F)
        os.system(cmd)
    
    def test_binfile2(self):
        """ Binfile with double + long data """

        data1d = pd.DataFrame(np.random.normal(size=(self.nrow, self.ncol)))

        data1l = pd.DataFrame(np.random.randint(0, 500, 
            size=(self.nrow, self.ncol)))

        data1 = pd.concat([data1d, data1l], axis=1)
        nc = data1.shape[1]
        cc = np.random.choice(range(nc), size=nc)
        data1 = data1.iloc[:, cc]

        ft = '%s/binfile_testdata2.bin'%self.FOUT        
        binfile.write_bin(data1, ft, 'test data')

        data2, comment = binfile.read_bin(ft)

        self.assertTrue(np.allclose(data1, data2))
        
        F = self.FOUT
        cmd = 'rm %s/*.bind %s/*.bins %s/*.binl' % (F, F, F)
        os.system(cmd)
    
    def test_binfile3(self):
        """ Binfile with double + long + string data """

        data1d = pd.DataFrame(np.random.normal(size=(self.nrow, self.ncol)))

        data1l = pd.DataFrame(np.random.randint(0, 500, 
                        size=(self.nrow, self.ncol)))

        strlength = 40
        i = np.random.randint(33, 123, size = strlength * self.nrow * self.ncol)
        i = i.reshape((self.nrow * self.ncol, strlength)).astype(np.int8)
        i = [s.tostring().decode('ascii') for s in i]
        data1s = pd.DataFrame(np.array(i).reshape((self.nrow, self.ncol)))
        data1s = data1s.apply(lambda x: 
            pd.Series([s[:np.random.randint(5, strlength)] for s in x]))

        data1 = pd.concat([data1d, data1l, data1s], axis=1)

        cc  = ['d%0.2d'%i for i in range(self.ncol)] 
        cc += ['l%0.2d'%i for i in range(self.ncol)] 
        cc += ['s%0.2d'%i for i in range(self.ncol)]
        data1.columns = cc

        nc = data1.shape[1]
        cc = np.random.choice(range(nc), size=nc)
        data1 = data1.iloc[:, cc]

        ft = '%s/binfile_testdata3.bin'%self.FOUT        
        binfile.write_bin(data1, ft, 'test data', strlength=strlength)
        data2, comment = binfile.read_bin(ft)
        data2.columns = data1.columns

       
        cc = [cn for cn in data1.columns if re.search('^(d|l)', cn)]
        self.assertTrue(np.allclose(data1[cc], data2[cc]))

        cc = [cn for cn in data1.columns if re.search('^s', cn)]
        self.assertTrue(np.all(data1[cc] == data2[cc]))

        F = self.FOUT
        cmd = 'rm %s/*.bind %s/*.bins %s/*.binl' % (F, F, F)
        os.system(cmd)
    
    def test_binfile4(self):
        """ Binfile from hym """

        ft = '%s/data/hym_test_iobin_3.bin'%self.FOUT        
        data, comment = binfile.read_bin(ft)

        self.assertTrue(data.shape == (1000, 6))

if __name__ == "__main__":
    unittest.main()
