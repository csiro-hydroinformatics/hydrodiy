import os, re
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
        sites, url = hykiwis.get_sites()
        self.assertTrue(not sites is None)
        self.assertTrue(isinstance(sites, pd.core.frame.DataFrame))


    def test_getstorages(self):
        ''' Test get storages '''
        storages = hykiwis.get_storages()
        self.assertEqual(storages.shape, (305, 18))

        cc = ['name', 'capacity[ML]', 'longitude', 'latitude']
        ck = np.all([cn in storages.columns for cn in cc])
        self.assertTrue(ck)


    def test_getattrs(self):
        ''' Test get attributes '''
        try:
            attrs, url = hykiwis.get_tsattrs('410001', 'daily_9am')

            if attrs is None:
                raise ValueError()

        except ValueError as err:
            if str(err).startswith('Request returns no data'):
                self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')
        attrs = attrs[0]
        self.assertTrue(isinstance(attrs, dict))
        self.assertEqual(attrs['station_name'], 'M/BIDGEE R @ WAGGA')
        self.assertEqual(attrs['ts_unitsymbol'], 'cumec')
        self.assertEqual(attrs['station_no'], '410001')


        try:
            attrs, url = hykiwis.get_tsattrs('613002', 'daily_9am')

            if attrs is None:
                raise ValueError()

        except ValueError as err:
            if str(err).startswith('Request returns no data'):
                self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')

        attrs = attrs[0]
        self.assertTrue(re.search('DINGO R', attrs['station_name'], \
                            re.IGNORECASE))
        self.assertEqual(attrs['ts_unitsymbol'], 'cumec')
        self.assertEqual(attrs['station_no'], '613002')


    def test_getattrs_multiple_series(self):
        ''' Test get attributes for sites with multiple series '''

        try:
            attrs, url = hykiwis.get_tsattrs('412010', 'as_stored')

            if attrs is None:
                raise ValueError()

        except ValueError as err:
            if str(err).startswith('Request returns no data'):
                self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')

        self.assertEqual(len(attrs), 6)


    def test_getdata_flow(self):
        ''' Test download flow data '''

        # Full download
        try:
            attrs, url = hykiwis.get_tsattrs('410001', 'daily_9am')

            if attrs is None:
                raise ValueError()

        except ValueError as err:
            if str(err).startswith('Request returns no data'):
                self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')

        attrs = attrs[0]
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


    def test_getdata_timeseries(self):
        ''' Test download timeseries data '''
        for ts_name in hykiwis.TS_NAMES:
            print(ts_name)
            try:
                attrs, url = hykiwis.get_tsattrs('410001', ts_name)

                if attrs is None:
                    raise ValueError()

            except ValueError as err:
                if str(err).startswith('Request returns no data'):
                    continue

            attrs = attrs[0]
            ts_data, url = hykiwis.get_data(attrs, '2010-01-01', '2010-12-31')
            print(len(ts_data))

            self.assertTrue(isinstance(ts_data, pd.core.series.Series))
            self.assertTrue(len(ts_data)>=1)


    def test_getdata_internal(self):
        ''' Test download data from internal.
            Skipped if there is no internal access to BOM Kiwis server
        '''
        if not hykiwis.has_internal_access():
            self.skipTest('No internal access available')

        # Full download
        attrs = None
        try:
            attrs, url = hykiwis.get_tsattrs('410001', 'daily_9am', \
                            external=False)

            if attrs is None:
                raise ValueError()

        except ValueError as err:
            if str(err).startswith('Request returns no data'):
                self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')

        if attrs is None:
            self.skipTest('Could not get ts attributes, '+\
                                'request returns no data')

        attrs = attrs[0]
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


    def test_getdata_storages(self):
        ''' Test download data - storage '''

        storages = hykiwis.get_storages()
        storages = storages[::40]

        for kiwisid, row in storages.iterrows():
            attrs = None
            try:
                attrs, url = hykiwis.get_tsattrs(kiwisid, 'daily_12pm')

                if attrs is None:
                    raise ValueError()

            except ValueError as err:
                if str(err).startswith('Request returns no data'):
                    self.skipTest('Could not get ts attributes, '+\
                                    'request returns no data')

            if attrs is None:
                self.skipTest('Could not get ts attributes, '+\
                                    'request returns no data')

            attrs = attrs[0]
            ts_data1, url = hykiwis.get_data(attrs)
            self.assertTrue(isinstance(ts_data1, pd.core.series.Series))


if __name__ == "__main__":
    unittest.main()
