import pytest

import pandas as pd

from hydrodiy.data import xval_periods

def test_period(allclose):
    start = "2010-01-01"
    end = "2010-12-31"
    xv = xval_periods.XvalPeriod(start, end)

    time = pd.date_range(xv.start_active, xv.end_active)
    assert len(time) == 365

    time = pd.date_range(xv.start_total, xv.end_total)
    expected = 365.25 * (1 + xv.warmup_years)
    assert allclose(len(time), expected, atol=1.)

    offset = pd.DateOffset(years=1)
    time = pd.date_range(xv.start_total - offset, xv.end_total + offset)
    idx = xv.index_active(time)
    assert idx.sum() == 365

    idx = xv.index_total(time)
    assert allclose(idx.sum(), expected, atol=1.)


@pytest.mark.parametrize("nper", [2, 5, 10, 20])
def test_periodset(nper, allclose):
    start = 1980
    end = 2010
    xv = xval_periods.XvalPeriodSet(start, end, nper)

    # Check printing works
    s = str(xv)

    # Check periods are of same length
    start = xv.periods["PER01"].start_total
    end = xv.periods[f"PER{nper:02d}"].end_total
    counts1 = pd.Series(0, index=pd.date_range(start, end))

    start = xv.periods["PER01"].start_active
    end = xv.periods[f"PER{nper:02d}"].end_active
    counts2 = pd.Series(0, index=pd.date_range(start, end))

    for iper, (pname, per) in enumerate(xv.periods.items()):
        d = per.nyears_active
        if iper == 0:
            d0 = d
        else:
            assert allclose(d, d0, atol=1e-2)

        idx = per.index_active(counts1.index)
        counts1.loc[idx] += 1

        idx = per.index_active(counts2.index)
        counts2.loc[idx] += 1

    # Check not overlap
    assert counts1.max() == 1
    assert counts2.max() == 1

    # Check no gap
    assert (counts2 == 0).sum() == 0

    # Check warmup duration
    nwarm = (counts1 == 0).sum()
    expected = xv.warmup_years * 365.25
    assert allclose(nwarm, expected, atol=1)
