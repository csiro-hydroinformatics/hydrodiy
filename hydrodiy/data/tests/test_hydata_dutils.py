import os
import time
import unittest
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta
import pandas as pd

from scipy.special import comb

from hydrodiy.data import dutils
import c_hydrodiy_data as chd


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
        idx = index >= datetime(1950, 2, 5)
        idx = idx & (index <= datetime(1950, 2, 7))
        u.loc[idx] = np.nan

        # -> 4 consecutive gaps
        idx = index >= datetime(1950, 3, 5)
        idx = idx & (index <= datetime(1950, 3, 8))
        u.loc[idx] = np.nan

        for d in [2, 5, 6, 8, 11, 20]:
            # -> 6 gaps
            dd = datetime(1950, 4, d)
            u.loc[dd] = np.nan

            # -> 6 gaps
            dd = datetime(1950, 5, d)
            u.loc[dd] = np.nan

        # -> one more gap
        dd = datetime(1950, 5, 22)
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

        kk = np.random.choice(range(nval), nval//10, replace=False)
        obs[kk] = np.nan
        obsm = obs.resample('MS', how=lambda x: np.sum(x.values))
        obsm2 = dutils.aggregate(aggindex, obs.values)

        idx = np.isnan(obsm.values)
        self.assertTrue(np.allclose(idx, np.isnan(obsm2)))
        self.assertTrue(np.allclose(obsm.values[~idx], obsm2[~idx]))

        obsm = obs.resample('MS', how='sum')
        obsm3 = dutils.aggregate(aggindex, obs.values, maxnan=31)
        self.assertTrue(np.allclose(obsm.values, obsm3))


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


    def test_monthly2daily_flat(self):
        ''' Test monthly2daily with flat disaggregation'''

        dates = pd.date_range('1990-01-01', '2000-12-31', freq='MS')
        nval = len(dates)
        se = pd.Series(np.exp(np.random.normal(size=nval)), index=dates)

        sed = dutils.monthly2daily(se)
        se2 = sed.resample('MS', how='sum')
        self.assertTrue(np.allclose(se.values, se2.values))


    def test_monthly2daily_cubic(self):
        ''' Test monthly2daily with cubic disaggregation'''

        dates = pd.date_range('1990-01-01', '2000-12-31', freq='MS')
        nval = len(dates)
        se = pd.Series(np.exp(np.random.normal(size=nval)), index=dates)

        sed = dutils.monthly2daily(se, interpolation='cubic')
        se2 = sed.resample('MS', how='sum')
        self.assertTrue(np.allclose(se.values, se2.values))


    def test_combi(self):
        ''' Test number of combinations '''

        for n in range(1, 65):
            for k in range(n):
                c1 = comb(n, k)
                c2 = chd.combi(n, k)
                ck = np.allclose(c1, c2)
                if c2>0:
                    self.assertTrue(ck)


    def test_var2h_hourly(self):
        ''' Test conversion to hourly for hourly data '''

        nval = 24*365*20
        dt = pd.date_range(start='1968-01-01', freq='H', periods=nval)
        se = pd.Series(np.arange(nval), index=dt)

        seh = dutils.var2h(se, display=True)
        expected = (se+se.shift(-1))/2

        idx = ~np.isnan(seh.values)
        self.assertTrue(np.allclose(seh.values[idx], \
                            expected[1:].values[idx]))


    def test_var2h_5min(self):
        ''' Test conversion to hourly for 10min data '''

        nval = 24 #*365*3
        dt = pd.date_range(start='1968-01-01', freq='5min', periods=nval*6)
        se = pd.Series(np.random.uniform(0, 1, size=len(dt)), index=dt)

        seh = dutils.var2h(se)

        # build expected
        nlag = 12
        for lag in range(nlag):
            d = (se.shift(-lag)+se.shift(-lag-1))/2
            expected = d if lag==0 else expected+d
        expected = expected/nlag
        expected = expected[expected.index.minute==0]

        # Run test
        idx = ~np.isnan(seh.values)
        self.assertTrue(np.allclose(seh.values[idx], \
                    expected.values[1:][idx]))


    def test_var2h_variable(self):
        ''' Test variable to hourly conversion '''

        nvalh =50
        varsec = []
        varvalues = []
        hvalues = []

        vprev = np.zeros(1)

        for i in range(nvalh):
            # Number of points
            n = np.random.randint(3, 20)

            # Timing
            dt = np.maximum(2, np.random.exponential(3600/(n-1), n-1))
            dt = (dt/np.sum(dt)*3599).astype(int)
            dt[-1] = 3600-np.sum(dt[:-1])

            # values
            v = np.random.exponential(10, n)
            v[0] = vprev[-1]

            # Compute hourly data
            agg = np.sum((v[1:]+v[:-1])/2*dt)/3600
            hvalues.append(agg)

            # Store values
            for k in range(n-1):
                t = np.sum(dt[:k])
                varsec.append(t+i*3600)
                varvalues.append(v[k])

            vprev = v.copy()

        # format data
        varvalues = np.array(varvalues)
        start = datetime(1970, 1, 1)
        dt = np.array([start+delta(seconds=int(s)) for s in varsec])
        se = pd.Series(varvalues, index=dt)

        # run function
        seh = dutils.var2h(se)

        # run test
        hvalues = np.array(hvalues)[1:-1]
        self.assertTrue(np.allclose(seh.values[:-1], \
                    hvalues))


    def test_hourly2daily(self):
        ''' Test conversion hourly to daily '''

        nval = 500
        dt = pd.date_range('1970-01-01', freq='H', periods=nval)
        se = pd.Series(np.random.uniform(0, 1, size=nval),  \
                index=dt)

        # Run function
        sed1 = dutils.hourly2daily(se)
        sed2 = dutils.hourly2daily(se, timestamp_end=False)
        sed3 = dutils.hourly2daily(se, start_hour=0)
        sed4 = dutils.hourly2daily(se, start_hour=0, timestamp_end=False)

        # expected values
        expected = np.zeros((len(sed1), 4))

        dt = se.index[se.index.hour==9]
        for it, t in enumerate(dt):
            idx = (se.index>=t) & (se.index<t+delta(days=1))
            if it<len(expected)-1:
                expected[it+1, 0] = se[idx].sum()
            expected[it, 1] = se[idx].sum()

        dt = se.index[se.index.hour==0]
        for it, t in enumerate(dt):
            idx = (se.index>=t) & (se.index<t+delta(days=1))
            if it<len(expected)-1:
                expected[it+1, 2] = se[idx].sum()
            expected[it, 3] = se[idx].sum()

        # Run tests
        self.assertTrue(np.allclose(sed1.values[1:-1], expected[1:-1, 0]))
        self.assertTrue(np.allclose(sed2.values[1:-1], expected[1:-1, 1]))
        self.assertTrue(np.allclose(sed3.values[1:-1], expected[1:-1, 2]))
        self.assertTrue(np.allclose(sed4.values[1:-1], expected[1:-1, 3]))


if __name__ == "__main__":
    unittest.main()
