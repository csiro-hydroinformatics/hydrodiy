import os, math
import time
import pytest
from pathlib import Path
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta
import pandas as pd

from scipy.special import comb

from hydrodiy.io import csv
from hydrodiy.data import dutils
from hydrodiy import has_c_module

# Try to import C code
if has_c_module("data", False):
    import c_hydrodiy_data as chd

# Test skip message
SKIPMESS = "c_hydrodiy_data module is not available. Please compile."

# Utility function to aggregate data
# using various version of pandas
def agg_d2m(x, fun="mean"):
    assert fun in ["mean", "sum", "max", "tail"]

    # Define aggregation function
    if fun == "mean":
        aggfun = lambda y: np.mean(y.values)
    elif fun == "sum":
        aggfun = lambda y: np.sum(y.values)
    elif fun == "max":
        aggfun = lambda y: np.nanmax(y.values)
    else:
        aggfun = lambda y: y.values[~np.isnan(y.values)][-1]

    # Run aggregation
    try:
        xa = x.resample("MS").apply(aggfun)

        # To handle case where old pandas syntax runs
        # but produces a single values
        if (len(xa) == 1 and len(np.unique(x.index.month)) > 1):
            raise ValueError

    except Exception:
        xa = x.resample("MS", how=aggfun)

    return xa


# Tests

def test_get_value_from_kwargs(allclose):
    kw = dict(firstarg=10, secondarg=100)
    v = dutils.get_value_from_kwargs(kw, "firstarg")
    assert v == 10
    assert kw == dict(secondarg=100)

    kw = dict(f=10)
    v = dutils.get_value_from_kwargs(kw, "firstarg", "f")
    assert v == 10
    assert kw == dict()

    kw = dict(firstarg=10)
    v = dutils.get_value_from_kwargs(kw, "bidule", "b", 100)
    assert v == 100
    assert kw == dict(firstarg=10)


def test_cast_scalar(allclose):
    """ Test scalar casts """
    x = 0.6
    y = 0.7
    ycast = dutils.cast(x, y)
    assert isinstance(ycast, type(x))
    assert allclose(ycast, 0.7)

    x = 0.6
    y = np.array(0.7) # 0d np.array
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, 0.7)

    x = np.float64(0.6) # numpy float type
    y = np.array(0.7) # 0d np.array
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, 0.7)

    x = 0.6
    y = np.array([0.7]) # 1d np.array
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, 0.7)

    x = 0.6
    y = 7
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, 7.)

    # we convert a float to int here
    x = 6
    y = np.array([0.7])
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, 0)


def test_cast_scalar_error(allclose):
    """ Test scalar cast errors """
    x = 6
    y = 7.
    msg = "Cannot cast"
    with pytest.raises(TypeError, match=msg):
        ycast = dutils.cast(x, y)

    x = 6
    y = np.array([0.7, 0.8])
    msg = "only "
    with pytest.raises(TypeError, match=msg):
        ycast = dutils.cast(x, y)


def test_cast_array(allclose):
    """ Test array cast """
    x = np.array([0.6, 0.7])
    y = np.array([0.7, 0.8])
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, y)

    x = np.array([0.6, 0.7])
    y = np.array([7, 8])
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, y)

    x = np.array([0.6, 0.7])
    y = 7.
    ycast = dutils.cast(x, y)
    assert (isinstance(ycast, type(x)))
    assert allclose(ycast, y)


def test_cast_array_error(allclose):
    """ Test scalar cast errors """
    x = np.array([6, 7])
    y = np.array([7., 8.])
    msg = "Cannot cast"
    with pytest.raises(TypeError, match=msg):
        ycast = dutils.cast(x, y)

    x = 6.
    y = np.array([7., 8.])
    msg = "only "
    with pytest.raises(TypeError, match=msg):
        ycast = dutils.cast(x, y)


def test_sequence_true(allclose):
    """ Test analysis of true sequence """
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
        assert allclose(duration, seq_len[::2])

        # Start with 0
        reps = np.zeros(20)
        reps[1::2] = 1
        seq = np.repeat(reps, seq_len)
        startend = dutils.sequence_true(seq)
        duration = startend[:, 1]-startend[:, 0]
        assert allclose(duration, seq_len[1::2])


def test_dayofyear(allclose):
    """ Test day of year """
    days = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    doy = dutils.dayofyear(days)
    assert allclose(doy, np.arange(1, 366))

    days = pd.date_range("2000-01-01", "2000-03-02", freq="D")
    doy = dutils.dayofyear(days)
    expected = np.append(np.arange(1, 61), [60, 61])
    assert allclose(doy, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_aggregate(allclose):
    dt = pd.date_range("1990-01-01", "2000-12-31")
    nval = len(dt)
    obs = pd.Series(np.random.uniform(0, 1, nval), \
            index=dt)

    aggindex = dt.year * 100 + dt.month

    obsm = agg_d2m(obs, fun="sum")
    obsm2 = dutils.aggregate(aggindex, obs.values)
    assert allclose(obsm.values, obsm2)

    obsm = agg_d2m(obs, fun="mean")
    obsm2 = dutils.aggregate(aggindex, obs.values, operator=1)
    assert allclose(obsm.values, obsm2)

    obsm = agg_d2m(obs, fun="max")
    obsm2 = dutils.aggregate(aggindex, obs.values, operator=2)
    assert allclose(obsm.values, obsm2)

    obsm = agg_d2m(obs, fun="tail")
    obsm2 = dutils.aggregate(aggindex, obs.values, operator=3)
    assert allclose(obsm.values, obsm2)

    kk = np.random.choice(range(nval), nval//10, replace=False)
    obs[kk] = np.nan
    obsm = obs.resample("MS").apply(lambda x: np.sum(x.values))
    obsm2 = dutils.aggregate(aggindex, obs.values)

    idx = np.isnan(obsm.values)
    assert np.all(idx == np.isnan(obsm2))
    assert allclose(obsm.values[~idx], obsm2[~idx])

    obsm = obs.resample("MS").sum()
    obsm3 = dutils.aggregate(aggindex, obs.values, maxnan=31)
    assert allclose(obsm.values, obsm3)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_aggregate_error(allclose):
    dt = pd.date_range("1990-01-01", "2000-12-31")
    nval = len(dt)
    obs = pd.Series(np.random.uniform(0, 1, nval),
                    index=dt)
    aggindex = dt.year * 100 + dt.month

    try:
        obsm = dutils.aggregate(aggindex[:100], obs.values)
    except ValueError as err:
        assert (str(err).startswith("Expected same length"))
    else:
        raise ValueError("Problem with error handling")


def test_lag(allclose):
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

            assert allclose(laggede, expected)

            if abs(lag)>0:
                assert np.all(np.isnan(na))


def test_monthly2daily_flat(allclose):
    dates = pd.date_range("1990-01-01", "2000-12-31", freq="MS")
    nval = len(dates)
    se = pd.Series(np.exp(np.random.normal(size=nval)), index=dates)

    sed = dutils.monthly2daily(se)
    se2 = agg_d2m(sed, fun="sum")
    assert allclose(se.values, se2.values)


def test_monthly2daily_cubic(allclose):
    dates = pd.date_range("1990-01-01", "2000-12-31", freq="MS")
    nval = len(dates)
    se = pd.Series(np.exp(np.random.normal(size=nval)), index=dates)

    sed = dutils.monthly2daily(se, interpolation="cubic", \
                            minthreshold=-np.inf)
    se2 = agg_d2m(sed, fun="sum")
    assert allclose(se.values, se2.values)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_combi(allclose):
    for n in range(1, 65):
        for k in range(n):
            c1 = comb(n, k)
            c2 = chd.combi(n, k)
            ck = np.allclose(c1, c2)
            if c2>0:
                assert ck


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_hourly(allclose):
    nval = 24*365*20
    dt = pd.date_range(start="1968-01-01", freq="h", periods=nval)
    se = pd.Series(np.arange(nval), index=dt)

    seh = dutils.var2h(se, display=True)
    expected = (se+se.shift(-1))/2

    idx = ~np.isnan(seh.values)
    assert allclose(seh.values[idx], \
                        expected[1:].values[idx])


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_5min(allclose):
    """ Test conversion to hourly for 10min data """
    nval = 24 #*365*3
    dt = pd.date_range(start="1968-01-01", freq="5min", \
                            periods=nval*6)
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
    assert allclose(seh.values[idx], \
                expected.values[1:][idx])


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_variable(allclose):
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
    assert allclose(seh.values[:-1], hvalues)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_longgap(allclose):
    nval = 6*20
    index = pd.date_range("1970-01-01", freq="10min", periods=nval)
    v = np.convolve(np.random.uniform(0, 100, size=nval), \
                                        np.ones(10))[:nval]
    se = pd.Series(v, index=index)
    se[30:60] = np.nan
    se = se[pd.notnull(se)]

    seh = dutils.var2h(se, maxgapsec=3600)
    assert np.all(np.isnan(seh.values[4:10]))
    assert np.all(~np.isnan(seh.values[:3]))
    assert np.all(~np.isnan(seh.values[10:-1]))


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_timezone(allclose):
    fts = Path(__file__).resolve().parent / "test_var2h.csv"
    se = pd.read_csv(fts, index_col=0, parse_dates=True).iloc[:, 0]
    seh = dutils.var2h(se, display=True)
    assert (len(seh) == 307832)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_halfhourly(allclose):
    nval = 24*365*20
    dt = pd.date_range(start="1968-01-01", freq="30min", periods=nval)
    se = pd.Series(np.arange(nval), index=dt)

    seh = dutils.var2h(se, display=True, nbsec_per_period=1800)
    expected = (se+se.shift(-1))/2

    idx = ~np.isnan(expected.values[2:])
    assert allclose(seh.values[:-1][idx], \
                        expected.values[2:][idx])


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_var2h_rainfall(allclose):
    values = [
        ["1/04/1989 7:50", 2.79], \
        ["1/04/1989 9:20",	3.09], \
        ["1/04/1989 9:44",	9.65], \
        ["1/04/1989 11:37",	3.45], \
        ["1/04/1989 13:06",	8.03], \
        ["1/04/1989 14:03",	25.78], \
        ["1/04/1989 14:21",	11.61], \
        ["1/04/1989 14:39",	32.65], \
        ["1/04/1989 14:55",	5.74], \
        ["1/04/1989 15:34",	50], \
        ["1/04/1989 15:51",	8.9], \
        ["1/04/1989 16:22",	41.1], \
        ["1/04/1989 16:34",	15.03], \
        ["1/04/1989 17:05",	4.22]
    ]
    se = pd.DataFrame(values)
    se.loc[:, 0] = pd.to_datetime(se.loc[:, 0], dayfirst=True)
    se = se.set_index(0).squeeze()

    seh = dutils.var2h(se, rainfall=True)
    t = "1989-04-01 14:00:00"
    assert allclose(seh.loc[t], 57.77, atol=1e-2)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_flathomogen(allclose):
    dt = pd.date_range("2000-01-10", "2000-04-05")
    a = pd.Series(np.random.uniform(0, 1, size=len(dt)), index=dt)
    a[15] = np.nan
    aggindex = a.index.year*100 + a.index.month
    af = dutils.flathomogen(aggindex, a.values, 1)

    assert (len(af) == len(a))

    expected = np.zeros(len(af))
    for ix in np.unique(aggindex):
        kk = aggindex == ix
        expected[kk] = np.nanmean(a[kk])
    expected[np.isnan(a.values)] = np.nan

    isok = ~np.isnan(af)
    assert allclose(af[isok], expected[isok])


def test_oz_timezone(allclose):
    """ Test time zones in Australia """
    tz = dutils.oz_timezone(147.83, -30.54)
    assert tz == "Australia/Sydney"

    tz = dutils.oz_timezone(147.17, -41.42)
    assert tz == "Australia/Sydney"

    tz = dutils.oz_timezone(142.11, -37.459)
    assert tz == "Australia/Sydney"

    tz = dutils.oz_timezone(141.5, -28.557)
    assert tz == "Australia/Brisbane"

    tz = dutils.oz_timezone(138.25, -25.425)
    assert tz == "Australia/Brisbane"

    tz = dutils.oz_timezone(137.59, -16.869)
    assert tz == "Australia/Darwin"

    tz = dutils.oz_timezone(140.09, -26.767)
    assert tz == "Australia/Adelaide"

    tz = dutils.oz_timezone(129.374, -29.823)
    assert tz == "Australia/Adelaide"

    tz = dutils.oz_timezone(128.451, -18.627)
    assert tz == "Australia/Perth"

    tz = dutils.oz_timezone(117.245, -34.181)
    assert tz == "Australia/Perth"


def test_water_year_end(allclose):
    t = pd.date_range("2000-01-01", "2010-12-31")
    a = 8
    for _ in range(100):
        e = np.random.uniform(-0.05, 0.05, size=len(t))
        q = pd.Series(e+(np.cos((t.month-a)*math.pi/6)+1)/2, \
                        index=t)

        wye = dutils.water_year_end(q)
        assert allclose(wye, 2)

    try:
        wy = dutils.water_year_end(q, 12)
    except AssertionError as err:
        assert (str(err).startswith("Expected convolve_window"))

    try:
        q = pd.Series(q.values)
        wy = dutils.water_year_end(q)
    except AssertionError as err:
        assert (str(err).startswith("Expected x with"))



def test_compute_aggindex(allclose):
    t = pd.date_range("2003-01-01", "2004-12-31")
    a = dutils.compute_aggindex(t, "AS")
    assert allclose(a[:365], 2003)
    assert allclose(a[365:], 2004)

    try:
        a = dutils.compute_aggindex(t, "AS-Bidule")
    except AssertionError as err:
        assert (str(err).startswith("Expected month"))

    a = dutils.compute_aggindex(t, "AS-JUN")
    assert allclose(a[:181], 2002)
    assert allclose(a[181:547], 2003)
    assert allclose(a[547:729], 2004)

    a = dutils.compute_aggindex(t, "AS-NOV")
    assert allclose(a[:334], 2002)
    assert allclose(a[334:700], 2003)
    assert allclose(a[700:729], 2004)

    m = dutils.compute_aggindex(t, "MS")
    assert allclose(m, t.year*100+t.month)

    t = pd.date_range("2003-01-01", "2004-12-31", freq="h")
    d = dutils.compute_aggindex(t, "D")
    assert np.unique(d).shape[0] == 731
    assert d[-1] == 20041231

    t = pd.date_range("2003-01-01 00:00:00", \
                    "2003-01-10 23:00:00", freq="min")
    h = dutils.compute_aggindex(t, "h")
    assert np.unique(h).shape[0] == 240
    assert h[-1] == 2003011023



