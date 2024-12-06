from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from hydrodiy.data import signatures
from hydrodiy import has_c_module

from hydrodiy.stat import transform
from hydrodiy.io import csv

FTEST = Path(__file__).resolve().parent
SKIPMESS = "c_hydrodiy_data module is not available. Please compile."

@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_eckhardt(allclose):
    fd = FTEST / "baseflow_RDF_EC.csv"
    data, _  = csv.read_csv(fd)
    flow = data.iloc[:, 2]
    bflow_expected = data.iloc[:, 3]

    bflow = signatures.eckhardt(flow, \
                tau=100,\
                thresh=0.95, \
                BFI_max = 0.80)

    assert allclose(bflow_expected, bflow)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_fdcslope(allclose):
    """ Test fdc slope computation """
    x = np.linspace(0, 1, 101)
    slp, qq = signatures.fdcslope(x, q1=90, q2=100, cst=0.5)
    assert allclose(slp, 1.01)
    assert allclose(qq, [0.9, 1])

    slplog, qqlog = signatures.fdcslope(x, q1=90, q2=100, cst=0.5, \
                trans=transform.Log())
    assert allclose(slplog, 1.063857825)
    assert allclose(qq, qqlog)


def test_fdcslope_error():
    """ Test fdc slope computation error """
    x = np.linspace(0, 1, 101)
    msg = "Expected q2 > q1"
    with pytest.raises(ValueError, match=msg):
        slp = signatures.fdcslope(x, q1=90, q2=80, cst=0.5)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_goue(allclose):
    dt = pd.date_range("2000-01-10", "2000-06-30")
    nt = len(dt)
    values = np.random.uniform(0, 1, nt)
    aggindex = dt.year*100 + dt.month
    gv = signatures.goue(aggindex, values)

    flat = values*0.
    for ix in np.unique(aggindex):
        kk = aggindex == ix
        flat[kk] = np.nanmean(values[kk])
    gv_expected = 1-np.sum((flat-values)**2)\
                    /np.sum((np.mean(values)-values)**2)

    assert allclose(gv, gv_expected)


def test_lag1corr():
    nval = 100
    values = np.random.uniform(0, 1, nval)
    rho = signatures.lag1corr(values)

