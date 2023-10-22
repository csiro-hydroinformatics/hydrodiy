import os
import pytest
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta as delta
import pandas as pd

from hydrodiy import has_c_module

if has_c_module("data", False):
    import c_hydrodiy_data as chd

# Fix seed
np.random.seed(42)

MONTHS = pd.date_range("1800-01-01", "2200-12-1", freq="MS")
DAYS = pd.date_range("1800-01-01", "2200-12-1", freq="5D")
SKIPMESS = "c_hydrodiy_data module is not available. Please compile."

@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_isleapyear():
    years = range(1800, 2200)
    isleap = pd.Series(years).apply(lambda x: f"{x}-02-29")
    isleap = pd.to_datetime(isleap, errors="coerce")
    isleap = pd.notnull(isleap)

    si = 0
    for y, i in zip(years, isleap):
        si += abs(int(i)-chd.isleapyear(y))
    assert si == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def testDAYSinmonth():
    sn = 0
    for m in MONTHS:
        nb = ((m+delta(months=1)-delta(days=1))-m).days + 1
        sn += abs(nb-chd.daysinmonth(m.year, m.month))
    assert sn == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_dayofyear():
    sd = 0
    for d in DAYS:
        nb = d.dayofyear

        # Correct for 29 Feb
        if chd.isleapyear(d.year) and d.month>=3:
            nb -= 1

        sd += abs(nb-chd.dayofyear(d.month, d.day))

    assert sd == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_add1month():
    sd = 0
    for d in DAYS:
        d2 = d+delta(months=1)
        dd2 = np.array([d2.year, d2.month, d2.day])

        dd = np.array([d.year, d.month, d.day]).astype(np.int32)
        chd.add1month(dd)

        err = abs(np.sum(dd-dd2))
        sd += err

    assert sd == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_add1day():
    sd = 0
    for d in DAYS:
        d2 = d+delta(days=1)
        dd2 = np.array([d2.year, d2.month, d2.day])

        dd = np.array([d.year, d.month, d.day]).astype(np.int32)
        chd.add1day(dd)

        err = abs(np.sum(dd-dd2))
        sd += err

    assert sd == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_comparedates():
    ntrial = 5000
    sd = 0
    for i in range(ntrial):
        k1 = np.random.choice(range(len(DAYS)), 1)
        d1 = DAYS[k1]
        dd1 = np.array([d1.year, d1.month, d1.day]).astype(np.int32)

        k2 = np.random.choice(range(len(DAYS)), 1)
        d2 = DAYS[k2]
        dd2 = np.array([d2.year, d2.month, d2.day]).astype(np.int32)

        diffa = 0
        if d1<d2:
            diffa = 1
        if d1>d2:
            diffa = -1
        diffb = chd.comparedates(dd1[:, 0], dd2[:, 0])
        err = abs(diffa-diffb)
        sd += err

    assert sd == 0


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_getdate():
    sd = 0
    for d in DAYS:
        day = d.year*1e4 + d.month*1e2 + d.day
        dt = np.array([d.year, d.month, d.day]).astype(np.int32)
        dt2 = dt*0
        chd.getdate(day, dt2)

        err = abs(np.sum(dt-dt2))
        sd += err

    assert sd == 0

