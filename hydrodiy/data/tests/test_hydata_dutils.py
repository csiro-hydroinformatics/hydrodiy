import os
import time
import unittest
import numpy as np
import datetime
import pandas as pd

from hydrodiy.data import dutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hydata)')
        self.dt = None # TODO

    def test_aggmonths(self):

        # Generate daily data with gaps
        index = pd.date_range('1950-01-01', '1950-12-31', freq='D')
        nval = len(index)
        val = np.random.uniform(size=nval)
        u = pd.Series(val, index=index)

        # -> 3 consecutive gaps
        idx = index >= datetime.datetime(1950, 2, 5)
        idx = idx & (index <= datetime.datetime(1950, 2, 7))
        u.loc[idx] = np.nan

        # -> 4 consecutive gaps
        idx = index >= datetime.datetime(1950, 3, 5)
        idx = idx & (index <= datetime.datetime(1950, 3, 8))
        u.loc[idx] = np.nan

        for d in [2, 5, 6, 8, 11, 20]:
            # -> 6 gaps
            dd = datetime.datetime(1950, 4, d)
            u.loc[dd] = np.nan

            # -> 6 gaps
            dd = datetime.datetime(1950, 5, d)
            u.loc[dd] = np.nan

        # -> one more gap
        dd = datetime.datetime(1950, 5, 22)
        u.loc[dd] = np.nan

        # Compute monthly and seasonal data
        out1 = dutils.aggmonths(u, nmonths=1)
        out2 = dutils.aggmonths(u, nmonths=3)

        # Test
        def _sum(x):
             return np.sum(x.values)

        expected1 = u.resample('MS', how=_sum)
        expected2 = out1 + out1.shift(-1) + out1.shift(-2)

        idxe = pd.notnull(expected1)
        self.assertTrue(np.allclose(expected1[idxe], out1[idxe]))

        idxe = pd.isnull(out1)
        idxo = (out1.index.month == 3) | (out1.index.month == 5)
        self.assertTrue(np.allclose(idxe, idxo))

        idxo = pd.notnull(out2)
        self.assertTrue(np.allclose(out2[idxo], expected2[idxo]))


    def test_atmpressure(self):
        alt = 0
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 101325.))

        alt = 100
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 100130.800974))

        alt = 200
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 98950.6765392))


    def test_aggregate(self):
        dt = pd.date_range('1990-01-01', '2000-12-31')
        nval = len(dt)
        obs = pd.Series(np.random.uniform(0, 1, nval), \
                index=dt)

        aggindex = dt.year * 100 + dt.month

        obsm = obs.resample('MS', how='sum')
        obsm2 = dutils.aggregate(aggindex, obs.values)
        self.assertTrue(np.allclose(obsm.values, obsm2))

        obsm = obs.resample('MS', how='mean')
        obsm2 = dutils.aggregate(aggindex, obs.values, oper=1)
        self.assertTrue(np.allclose(obsm.values, obsm2))

        kk = np.random.choice(range(nval), nval/10, replace=False)
        obs[kk] = np.nan
        obsm = obs.resample('MS', how=lambda x: np.sum(x.values))
        obsm2 = dutils.aggregate(aggindex, obs.values)

        idx = np.isnan(obsm.values)
        self.assertTrue(np.allclose(idx, np.isnan(obsm2)))
        self.assertTrue(np.allclose(obsm.values[~idx], obsm2[~idx]))

        obsm = obs.resample('MS', how='sum')
        obsm3 = dutils.aggregate(aggindex, obs.values, maxnan=31)
        self.assertTrue(np.allclose(obsm.values, obsm3))

        # Compare timing with  pandas
        dt = pd.date_range('1700-01-01', '2100-12-31')
        nval = len(dt)
        obs = pd.Series(np.random.uniform(0, 1, nval), \
                index=dt)

        aggindex = dt.year * 100 + dt.month

        t0 = time.time()
        obsm = obs.resample('MS', how='sum')
        t1 = time.time()
        obsm2 = dutils.aggregate(aggindex, obs.values)
        t2 = time.time()
        self.assertTrue(t1-t0 > 80*(t2-t1))


    def test_lag(self):
        ''' Test lag for 1d data'''
        size = [20, 10, 30, 4]
        for ndim in  range(1, 4):
            data = np.random.normal(size=size[:ndim])

            for lag in range(-5, 6):
                lagged = dutils.lag(data, lag)
                if lag > 0:
                    expected = data[:-lag]
                    laggede = lagged[lag:]
                    na = lagged[:lag]
                elif lag < 0:
                    expected = data[-lag:]
                    laggede = lagged[:lag]
                    na = lagged[lag:]
                else:
                    expected = data
                    laggede = lagged

                self.assertTrue(np.allclose(laggede, expected))

                if abs(lag)>0:
                    self.assertTrue(np.all(np.isnan(na)))


if __name__ == "__main__":
    unittest.main()
