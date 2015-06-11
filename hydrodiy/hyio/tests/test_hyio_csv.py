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

        data, comment = csv.read_csv(fcsv, has_colnames=False)

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

    def test_read_csv4(self):

        fcsv = '%s/climate.csv.gz'%self.FOUT

        data, comment = csv.read_csv(fcsv, 
                parse_dates=[''], index_col=0)

        self.assertTrue(len(comment) == 8)
        self.assertTrue(comment['written_on'] == '2014-08-12 12:41')

        d = data.index[0]
        self.assertTrue(isinstance(d, pd.tslib.Timestamp))

    def test_write_csv(self):

        fcsv = '%s/testwrite.csv'%self.FOUT

        nval = 100
        nc = 5
        idx = pd.date_range('1990-01-01', periods=nval, freq='D')
        df1 = pd.DataFrame(np.random.normal(size=(nval, nc)), index=idx)

        csv.write_csv(df1, fcsv, ['Random data'], 
                os.path.abspath(__file__),
                index=True)

        df2, comment = csv.read_csv(fcsv, 
                parse_dates=[''], index_col=0)

        self.assertTrue(int(comment['nrow']) == nval)
        self.assertTrue(int(comment['ncol']) == nc)

        d = df2.index[0]
        self.assertTrue(isinstance(d, pd.tslib.Timestamp))

        self.assertTrue(np.allclose(df1, df2))


if __name__ == "__main__":
    unittest.main()
