import os
import unittest
import numpy as np
import pandas as pd
import tarfile
import zipfile

from hydrodiy.io import csv

class CsvTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CsvTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_read_csv1(self):
        fcsv = '%s/states_centroids.csv.gz'%self.ftest
        data, comment = csv.read_csv(fcsv)
        st = pd.Series(['ACT', 'NSW', 'NT', 'QLD', 'SA',
                            'TAS', 'VIC', 'WA'])
        self.assertTrue(all(data['state']==st))


    def test_read_csv2(self):
        fcsv = '%s/states_centroids_noheader.csv'%self.ftest
        data, comment = csv.read_csv(fcsv, has_colnames=False)
        st = pd.Series(['ACT', 'NSW', 'NT', 'QLD', 'SA',
                            'TAS', 'VIC', 'WA'])
        self.assertTrue(all(data[0]==st))


    def test_read_csv3(self):
        fcsv = '%s/multiindex.csv'%self.ftest
        data, comment = csv.read_csv(fcsv)

        cols =['metric', 'runoff_rank',
                'logsinh-likelihood', 'logsinh-shapirotest',
                'yeojohnson-likelihood', 'yeojohnson-shapirotest']

        self.assertTrue(all(data.columns==cols))


    def test_read_csv4(self):
        fcsv = '%s/climate.csv'%self.ftest

        data, comment = csv.read_csv(fcsv,
                parse_dates=[''], index_col=0)

        self.assertTrue(len(comment) == 8)
        self.assertTrue(comment['written_on'] == '2014-08-12 12:41')

        d = data.index[0]
        self.assertTrue(isinstance(d, pd.tslib.Timestamp))


    def test_read_csv5(self):
        fcsv = '%s/207004_monthly_total_01.csv'%self.ftest
        data, comment = csv.read_csv(fcsv,
                parse_dates=True, index_col=0)


    def test_read_csv_latin(self):
        fcsv = '%s/latin_1.zip'%self.ftest
        try:
            data, comment = csv.read_csv(fcsv,
                    comment='#')
        except UnicodeDecodeError as err:
            pass

        data, comment = csv.read_csv(fcsv,
                comment='#', encoding='latin_1')
        self.assertTrue(np.allclose(data.iloc[:, 1:4].values, -99))


    def test_write_csv1(self):
        nval = 100
        nc = 5
        idx = pd.date_range('1990-01-01', periods=nval, freq='D')
        df1 = pd.DataFrame(np.random.normal(size=(nval, nc)), index=idx)

        fcsv1 = '%s/testwrite1.csv'%self.ftest
        csv.write_csv(df1, fcsv1, 'Random data',
                os.path.abspath(__file__),
                write_index=True)

        fcsv2 = '%s/testwrite2.csv'%self.ftest
        csv.write_csv(df1, fcsv2, 'Random data',
                os.path.abspath(__file__),
                float_format=None,
                write_index=True)

        df1exp, comment = csv.read_csv(fcsv1,
                parse_dates=[''], index_col=0)

        df2exp, comment = csv.read_csv(fcsv2,
                parse_dates=[''], index_col=0)

        self.assertTrue(int(comment['nrow']) == nval)
        self.assertTrue(int(comment['ncol']) == nc)

        d = df1exp.index[0]
        self.assertTrue(isinstance(d, pd.tslib.Timestamp))

        self.assertTrue(np.allclose(np.round(df1.values, 3), df1exp))
        self.assertTrue(np.allclose(df1, df2exp))


    def test_write_csv2(self):
        nval = 100
        nc = 5
        idx = pd.date_range('1990-01-01', periods=nval, freq='D')
        df1 = pd.DataFrame(np.random.normal(size=(nval, nc)), index=idx)

        comment1 = {'co1':'comment', 'co2':'comment 2'}
        fcsv = '%s/testwrite.csv'%self.ftest
        csv.write_csv(df1, fcsv, comment1,
                author='toto',
                source_file=os.path.abspath(__file__),
                write_index=True)

        df2, comment2 = csv.read_csv(fcsv,
                parse_dates=[''], index_col=0)

        self.assertTrue(comment1['co1'] == comment2['co1'])
        self.assertTrue(comment1['co2'] == comment2['co2'])
        self.assertTrue('toto' == comment2['author'])
        self.assertTrue(os.path.abspath(__file__) == comment2['source_file'])


    def test_write_csv3(self):
        nval = 100
        idx = pd.date_range('1990-01-01', periods=nval, freq='D')
        ds1 = pd.Series(np.random.normal(size=nval), index=idx)

        fcsv1 = '%s/testwrite3.csv'%self.ftest
        csv.write_csv(ds1, fcsv1, 'Random data',
                os.path.abspath(__file__),
                write_index=True)

        ds1exp, comment = csv.read_csv(fcsv1,
                parse_dates=[''], index_col=0)
        ds1exp = ds1exp.squeeze()

        self.assertTrue(np.allclose(ds1.round(3), ds1exp))


    def test_read_write_zip(self):
        # Generate data
        df = {}
        for i in range(4):
            df['test_{0:02d}/test_{0}.csv'.format(i)] = \
                    pd.DataFrame(np.random.normal(size=(100, 4)))

        # Write data to archive
        farc = os.path.join(self.ftest, 'test_archive.zip')
        with zipfile.ZipFile(farc, 'w') as arc:
            for k in df:
                # Add file to tar with a directory structure
                csv.write_csv(df[k],
                    filename=k,
                    comment='test '+str(i),
                    archive=arc,
                    float_format='%0.20f',
                    source_file=os.path.abspath(__file__))

        # Read it back and compare
        with zipfile.ZipFile(farc, 'r') as arc:
            for k in df:
                df2, _ = csv.read_csv(k, archive=arc)
                self.assertTrue(np.allclose(df[k].values, df2.values))


if __name__ == "__main__":
    unittest.main()
