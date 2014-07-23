import os
import unittest
import numpy as np
import pandas as pd
from hyio import csv

class CsvTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CsvTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_read_csv1(self):
        fcsv = '%s/states_centroids.csv.gz'%self.FOUT
        data, comment = csv.read_csv(fcsv)
         
        st = pd.Series(['ACT', 'NSW', 'NT', 'QLD', 'SA', 
                            'TAS', 'VIC', 'WA'])
        self.assertTrue(all(data['state']==st))
        
    def test_read_csv2(self):
        fcsv = '%s/states_centroids_noheader.csv'%self.FOUT
        data, comment = csv.read_csv(fcsv, has_header=False)
        st = pd.Series(['ACT', 'NSW', 'NT', 'QLD', 'SA', 
                            'TAS', 'VIC', 'WA'])
        self.assertTrue(all(data[0]==st))
        
    def test_read_csv3(self):
        fcsv = '%s/multiindex.csv'%self.FOUT
        data, comment = csv.read_csv(fcsv)

        cols =['metric', 'runoff_rank',
                'logsinh-likelihood', 'logsinh-shapirotest',
                'yeojohnson-likelihood', 'yeojohnson-shapirotest']

        self.assertTrue(all(data.columns==cols))

if __name__ == "__main__":
    unittest.main()
