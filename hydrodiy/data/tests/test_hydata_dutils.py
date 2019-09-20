import os
import time
import unittest
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta
import pandas as pd

from scipy.special import comb

from hydrodiy.io import csv
from hydrodiy.data import dutils
from hydrodiy.data.dutils import HAS_C_DATA_MODULE

# Try to import C code
if HAS_C_DATA_MODULE:
    import c_hydrodiy_data as chd

# Utility function to aggregate data
# using various version of pandas
def agg_d2m(x, fun='mean'):
    assert fun in ['mean', 'sum']

    # Define aggregation function
    if fun == 'mean':
        aggfun = lambda y: np.mean(y.values)
    else:
        aggfun = lambda y: np.sum(y.values)

    # Run aggregation
    try:
        xa = x.resample('MS').apply(aggfun)

        # To handle case where old pandas syntax runs
        # but produces a single values
        if (len(xa) == 1 and len(np.unique(x.index.month)) > 1):
            raise ValueError

    except Exception:
        xa = x.resample('MS', how=aggfun)

    return xa


# Tests
class CastTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CastTestCase (hystat)')

    def test_cast_scalar(self):
        ''' Test scalar casts '''
        x = 0.6
        y = 0.7
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 0.7))

        x = 0.6
        y = np.array(0.7) # 0d np.array
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 0.7))

        x = np.float64(0.6) # numpy float type
        y = np.array(0.7) # 0d np.array
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 0.7))

        x = 0.6
        y = np.array([0.7]) # 1d np.array
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 0.7))

        x = 0.6
        y = 7
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 7.))

        # we convert a float to int here
        x = 6
        y = np.array([0.7])
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.isclose(ycast, 0))


    def test_cast_scalar_error(self):
        ''' Test scalar cast errors '''
        x = 6
        y = 7.
        try:
            ycast = dutils.cast(x, y)
        except TypeError as err:
            self.assertTrue(str(err).startswith('Cannot cast'))
        else:
            raise ValueError('Problem in error handling')

        x = 6
        y = np.array([0.7, 0.8])
        try:
            ycast = dutils.cast(x, y)
        except TypeError as err:
            self.assertTrue(str(err).startswith('only length-1 arrays'))
        else:
            raise ValueError('Problem in error handling')


    def test_cast_array(self):
        ''' Test array cast '''
        x = np.array([0.6, 0.7])
        y = np.array([0.7, 0.8])
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.allclose(ycast, y))

        x = np.array([0.6, 0.7])
        y = np.array([7, 8])
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.allclose(ycast, y))

        x = np.array([0.6, 0.7])
        y = 7.
        ycast = dutils.cast(x, y)
        self.assertTrue(isinstance(ycast, type(x)))
        self.assertTrue(np.allclose(ycast, y))



    def test_cast_array_error(self):
        ''' Test scalar cast errors '''
        x = np.array([6, 7])
        y = np.array([7., 8.])
        try:
            ycast = dutils.cast(x, y)

        except TypeError as err:
            self.assertTrue(str(err).startswith('Cannot cast'))
        else:
            raise ValueError('Problem in error handling')

        x = 6.
        y = np.array([7., 8.])
        try:
            ycast = dutils.cast(x, y)

        except TypeError as err:
            self.assertTrue(str(err).startswith('only '))
        else:
            raise ValueError('Problem in error handling')



class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hydata)')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_sub(self):
        ''' Test replacing with sub '''
        # Basic test
        text0 = 'this is a weird sentence'
        text1 = dutils.sub(text0, {'this': 'that', 'weird':'cool'})
        self.assertEqual(text1, 'that is a cool sentence')

        # With special characters in the source text
        text0 = 'this is a [*.*) // weird sentence'
        text1 = dutils.sub(text0, {'this': 'that', 'weird':'cool'})
        self.assertEqual(text1, 'that is a [*.*) // cool sentence')

        # With special characters in the source and replacement text
        text0 = 'this is a [*.*) // weird sentence'
        text1 = dutils.sub(text0, {'this': 'that.*', 'weird':'cool$'})
        self.assertEqual(text1, 'that.* is a [*.*) // cool$ sentence')

        text0 = 'this_ab_'
        text1 = dutils.sub(text0, {'_$': '', '_(?<!s)': ' '})
        self.assertEqual(text1, 'this ab')


    def test_sequence_true(self):
        ''' Test analysis of true sequence '''

        nrepeat = 100
        nseq = 10
        for irepeat in range(nrepeat):
            seq_len = np.random.randint(5, 20, size=20)

            # Start with 1
            reps = np.zeros(20)
            reps[::2] = 1
            seq = np.repeat(reps, seq_len)
            startend = dutils.sequence_true(seq)
            duration = startend[:, 1]-startend[:, 0]
            self.assertTrue(np.allclose(duration, seq_len[::2]))

            # Start with 0
            reps = np.zeros(20)
            reps[1::2] = 1
            seq = np.repeat(reps, seq_len)
            startend = dutils.sequence_true(seq)
            duration = startend[:, 1]-startend[:, 0]
            self.assertTrue(np.allclose(duration, seq_len[1::2]))


    def test_dayofyear(self):
        ''' Test day of year '''
        days = pd.date_range('2001-01-01', '2001-12-31', freq='D')
        doy = dutils.dayofyear(days)
        self.assertTrue(np.allclose(doy, np.arange(1, 366)))

        days = pd.date_range('2000-01-01', '2000-03-02', freq='D')
        doy = dutils.dayofyear(days)
        expected = np.append(np.arange(1, 61), [60, 61])
        self.assertTrue(np.allclose(doy, expected))


    def test_aggmonths(self):
        ''' Test month aggregation '''
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
        expected1 = agg_d2m(u, 'sum')
        expected2 = out1 + out1.shift(-1) + out1.shift(-2)

        idxo = pd.notnull(expected1)
        self.assertTrue(np.allclose(out1[idxo], expected1[idxo]))

        idxo = pd.notnull(expected2)
        self.assertTrue(np.allclose(out2[idxo], expected2[idxo]))


    def test_atmpressure(self):
        ''' Test computation of atmospheric pressure '''
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
        ''' Test aggregation '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

        dt = pd.date_range('1990-01-01', '2000-12-31')
        nval = len(dt)
        obs = pd.Series(np.random.uniform(0, 1, nval), \
                index=dt)

        aggindex = dt.year * 100 + dt.month

        obsm = agg_d2m(obs, fun='sum')
        obsm2 = dutils.aggregate(aggindex, obs.values)
        self.assertTrue(np.allclose(obsm.values, obsm2))

        obsm = agg_d2m(obs, fun='mean')
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


    def test_aggregate_error(self):
        ''' Test aggregation error '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

        dt = pd.date_range('1990-01-01', '2000-12-31')
        nval = len(dt)
        obs = pd.Series(np.random.uniform(0, 1, nval), \
                index=dt)

        aggindex = dt.year * 100 + dt.month

        try:
            obsm = dutils.aggregate(aggindex[:100], obs.values)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected same length'))
        else:
            raise ValueError('Problem with error handling')


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
        se2 = agg_d2m(sed, fun='sum')
        self.assertTrue(np.allclose(se.values, se2.values))


    def test_monthly2daily_cubic(self):
        ''' Test monthly2daily with cubic disaggregation'''
        dates = pd.date_range('1990-01-01', '2000-12-31', freq='MS')
        nval = len(dates)
        se = pd.Series(np.exp(np.random.normal(size=nval)), index=dates)

        sed = dutils.monthly2daily(se, interpolation='cubic', \
                                minthreshold=-np.inf)
        se2 = agg_d2m(sed, fun='sum')
        self.assertTrue(np.allclose(se.values, se2.values))


    def test_combi(self):
        ''' Test number of combinations '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

        for n in range(1, 65):
            for k in range(n):
                c1 = comb(n, k)
                c2 = chd.combi(n, k)
                ck = np.allclose(c1, c2)
                if c2>0:
                    self.assertTrue(ck)


    def test_var2h_hourly(self):
        ''' Test conversion to hourly for hourly data '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

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
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

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
        ''' Test variable to hourly conversion by comparing with python
        algorithm '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

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


    def test_var2h_longgap(self):
        ''' Test variable to hourly conversion and apply to dataset '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

        nval = 6*20
        index = pd.date_range('1970-01-01', freq='10min', periods=nval)
        v = np.convolve(np.random.uniform(0, 100, size=nval), \
                                            np.ones(10))[:nval]
        se = pd.Series(v, index=index)
        se[30:60] = np.nan
        se = se[pd.notnull(se)]

        seh = dutils.var2h(se, maxgapsec=3600)
        self.assertTrue(np.all(np.isnan(seh.values[4:10])))
        self.assertTrue(np.all(~np.isnan(seh.values[:3])))
        self.assertTrue(np.all(~np.isnan(seh.values[10:-1])))


    def test_flathomogen(self):
        ''' Test flat disaggregation '''
        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C module c_hydrodiy_data')

        dt = pd.date_range('2000-01-10', '2000-04-05')
        a = pd.Series(np.random.uniform(0, 1, size=len(dt)), index=dt)
        a[15] = np.nan
        aggindex = a.index.year*100 + a.index.month
        af = dutils.flathomogen(aggindex, a.values, 1)

        self.assertTrue(len(af) == len(a))

        expected = np.zeros(len(af))
        for ix in np.unique(aggindex):
            kk = aggindex == ix
            expected[kk] = np.nanmean(a[kk])
        expected[np.isnan(a.values)] = np.nan

        isok = ~np.isnan(af)
        self.assertTrue(np.allclose(af[isok], expected[isok]))


if __name__ == "__main__":
    unittest.main()
