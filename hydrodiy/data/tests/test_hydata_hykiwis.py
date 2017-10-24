import os
import unittest
import numpy as np
from datetime import datetime
import pandas as pd

from hydrodiy.data import hykiwis

class HyKiwisTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyKiwisTestCase (hydata)')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_getsites(self):
        ''' Test get sites '''
        sites, url = hykiwis.get_sites(external=False)
        self.assertTrue(not sites is None)
        self.assertTrue(isinstance(sites, pd.core.frame.DataFrame))
        self.assertTrue(sites.shape[0]>25000)


    def test_getattrs(self):
        ''' Test get attributes '''

        attrs, url = hykiwis.get_tsattrs('410001')
        self.assertTrue(isinstance(attrs, dict))
        self.assertEqual(attrs['station_name'], 'M/BIDGEE R @ WAGGA')
        self.assertEqual(attrs['ts_unitsymbol'], 'Ml/d')
        self.assertEqual(attrs['station_no'], '410001')

        attrs, url = hykiwis.get_tsattrs('613002')
        self.assertEqual(attrs['station_name'], 'DINGO ROAD')
        self.assertEqual(attrs['ts_unitsymbol'], 'Ml/d')
        self.assertEqual(attrs['station_no'], '613002')


    def test_getdata(self):
        ''' Test download data '''

        # Full download
        attrs, url = hykiwis.get_tsattrs('410001')
        ts_data1, url = hykiwis.get_data(attrs)
        self.assertTrue(isinstance(ts_data1, pd.core.series.Series))
        self.assertEqual(ts_data1.index[0].year, hykiwis.START_YEAR)

        # Restricted download
        start = '2001-01-01'+attrs['to'][10:]
        end = '2001-01-05'+attrs['to'][10:]
        ts_data2, url = hykiwis.get_data(attrs, start, end)

        index = ts_data2.index + pd.tseries.offsets.DateOffset(hours=-9)
        expected = pd.date_range('2001-01-01', '2001-01-05')

        v1 = index.values.astype(float)
        v2 = expected.values.astype(float)
        self.assertTrue(np.allclose(v1, v2))


    def test_getdata_internal(self):
        ''' Test download data from internal '''

        # Full download
        attrs, url = hykiwis.get_tsattrs('410001', external=False)
        ts_data1, url = hykiwis.get_data(attrs, external=False)
        self.assertTrue(isinstance(ts_data1, pd.core.series.Series))
        self.assertEqual(ts_data1.index[0].year, hykiwis.START_YEAR)

        # Restricted download
        start = '2001-01-01'+attrs['to'][10:]
        end = '2001-01-05'+attrs['to'][10:]
        ts_data2, url = hykiwis.get_data(attrs, start, end, external=False)

        index = ts_data2.index + pd.tseries.offsets.DateOffset(hours=-9)
        expected = pd.date_range('2001-01-01', '2001-01-05')

        v1 = index.values.astype(float)
        v2 = expected.values.astype(float)
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    unittest.main()
