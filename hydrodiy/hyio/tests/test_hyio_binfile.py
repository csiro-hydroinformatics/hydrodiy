import os
import re
import unittest
import numpy as np
import pandas as pd

from hyio import binfile
from hydata import dutils

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

        start = 19950310000000

        # Writes data
        ft = '%s/binfile_testdata1.bin'%self.FOUT        
        binfile.write_bin(data1, ft, comment = 'test data')

        # Reads it back
        data2, sl, comment = binfile.read_bin(ft)

        self.assertTrue(np.allclose(data1, data2))

        F = self.FOUT
        cmd = 'rm %s/*.bind %s/*.bins %s/*.binl %s/*.binh' % (F, F, F, F)
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

        # Writes data
        ft = '%s/binfile_testdata2.bin'%self.FOUT        
        binfile.write_bin(data1, ft, 'test data')

        # Reads it back
        data2, sl, comment = binfile.read_bin(ft)

        self.assertTrue(np.allclose(data1, data2))
        
        F = self.FOUT
        cmd = 'rm %s/*.bind %s/*.bins %s/*.binl %s/*.binh' % (F, F, F, F)
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
       cc = np.random.choice(cc, size=nc)
       data1 = data1.loc[:, cc]

       # Write data
       ft = '%s/binfile_testdata3.bin'%self.FOUT        
       binfile.write_bin(data1, ft, 'test data', strlength=strlength)

       # Reads it back
       data2, sl, comment = binfile.read_bin(ft)
       data2.columns = data1.columns
      
       # Test equality
       cc = [cn for cn in data1.columns if re.search('^(d|l)', cn)]
       self.assertTrue(np.allclose(data1[cc], data2[cc]))

       cc = [cn for cn in data1.columns if re.search('^s', cn)]
       self.assertTrue(np.all(data1[cc] == data2[cc]))

       F = self.FOUT
       cmd = 'rm %s/*.bind %s/*.bins %s/*.binl %s/*.binh' % (F, F, F, F)
       os.system(cmd)
    
    def test_binfile4(self):
       """ Binfile with double + long + datetime data """

       data1d = pd.DataFrame(np.random.normal(size=(self.nrow, self.ncol)))

       data1l = pd.DataFrame(np.random.randint(0, 500, 
                       size=(self.nrow, self.ncol)))

       data1t = pd.date_range('1950-01-01', freq='D', 
                       periods= self.nrow * self.ncol).values
       data1t = pd.DataFrame(data1t.reshape((self.nrow, self.ncol)))

       data1 = pd.concat([data1d, data1l, data1t], axis=1)

       cc  = ['d%0.2d'%i for i in range(self.ncol)] 
       cc += ['l%0.2d'%i for i in range(self.ncol)] 
       cc += ['t%0.2d'%i for i in range(self.ncol)]
       data1.columns = cc

       nc = data1.shape[1]
       cc = list(np.random.choice(cc, size=nc))
       data1 = data1.loc[:, cc]

       # Write data
       ft = '%s/binfile_testdata4.bin'%self.FOUT        
       binfile.write_bin(data1, ft, 'test data')

       # Reads it back
       data2, sl, comment = binfile.read_bin(ft)
       data2.columns = data1.columns
      
       # Test equality
       cc = [cn for cn in data1.columns if re.search('^(d|l)', cn)]
       self.assertTrue(np.allclose(data1[cc], data2[cc]))

       cc = [cn for cn in data1.columns if re.search('^t', cn)]
       dd = data2[cc].apply(lambda x: np.array([dutils.osec2time(v) for v in x]))
       d = (data1[cc] - dd).astype(int)
       self.assertTrue(np.all(d == 0))

       F = self.FOUT
       cmd = 'rm %s/*.bind %s/*.bins %s/*.binl %s/*.binh' % (F, F, F, F)
       os.system(cmd)
    
    def test_binfile5(self):
        """ Binfile from hym """

        ft = '%s/data/hym_test_iobin_999.bin'%self.FOUT        
        data, sl, comment = binfile.read_bin(ft)

        self.assertTrue(data.shape == (26, 3))
        
        self.assertTrue(np.allclose(data.iloc[:,0], np.arange(26)))
        self.assertTrue(np.allclose(data.iloc[:,1], np.arange(26)))

        s = pd.Series([chr(i) for i in range(32, 58)])
        self.assertTrue(np.all(data.iloc[:,2] == s))


if __name__ == "__main__":
    unittest.main()
