import os
import pytest
import numpy as np
import pandas as pd
from hydrodiy.data import qualitycontrol as qc
from hydrodiy import has_c_module

np.random.seed(0)

SKIPMESS = "c_hydrodiy_data module is not available. Please compile."


def test_ismisscens(allclose):
    ''' Test detection of missing and censored data '''
    # Basic test
    a = np.ones(10)
    a[0] = np.nan
    a[1] = -1
    icens = qc.ismisscens(a)
    assert allclose(icens, [0, 1]+[2]*8)

    # Dimensions
    a = np.ones((10, 1))
    icens = qc.ismisscens(a)
    assert allclose(icens, [2]*10)

    a = np.ones((10, 2))
    icens = qc.ismisscens(a)
    assert allclose(icens, [8]*10)

    a[0, 1] = -10
    a[1, 1] = np.nan
    icens = qc.ismisscens(a)
    # first column valid = 2
    # second column censored = 1
    # total = 2+3*1 = 5
    assert icens[0]==5

    # first column valid = 2
    # second column missing = 0
    # total = 2+3*0 = 2
    assert icens[1]==2

    assert allclose(icens[2:], [8]*8)

    a = np.ones((10, 2, 4))
    msg = "Expected 1d or 2d data"
    with pytest.raises(ValueError, match=msg):
        icens = qc.ismisscens(a)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_error():
    nval = 20
    data = np.random.normal(size=nval)

    msg = "Expected npoints"
    with pytest.raises(ValueError, match=msg):
        status = qc.islinear(data, npoints=0)

    msg = "Expected tol"
    with pytest.raises(ValueError, match=msg):
        status = qc.islinear(data, tol=1e-11)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_linspace(allclose):
    nval = 20
    data = np.random.normal(size=nval)
    data[3:17] = np.linspace(0, 1, 14)

    status = qc.islinear(data, npoints=1)

    expected = np.zeros(data.shape[0])
    expected[3:17] = 1

    assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_constant(allclose):
    nval = 20
    data = np.random.normal(size=nval)
    data[3:17] = 100

    status = qc.islinear(data, npoints=1)

    expected = np.zeros(data.shape[0])
    expected[3:17] = 2

    assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_nan(allclose):
    nval = 20
    data = np.random.normal(size=nval)
    data[3:17] = np.linspace(0, 1, 14)
    data[12:16] = np.nan

    status = qc.islinear(data, npoints=1)

    expected = np.zeros(data.shape[0])
    expected[3:12] = 1

    assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_linspace_npoints(allclose):
    nval = 30
    data = np.random.normal(size=nval)

    i1 = 10
    i2 = 20
    idxlin = np.arange(i1, i2+1)
    data[idxlin] = np.linspace(0, 1, 11)

    for npoints in range(2, 5):
        status = qc.islinear(data, npoints)

        expected = np.zeros(data.shape[0])
        expected[i1:i2+1] = 1

        assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_zeros(allclose):
    nval = 20

    data = np.random.normal(size=nval)
    data[5:9] = 0.
    status = qc.islinear(data, npoints=1, thresh=data.min()-1)

    expected = np.zeros(data.shape[0])
    expected[5:9] = 2
    assert allclose(status, expected)

    status = qc.islinear(data, npoints=1, thresh=0.)
    expected = np.array([False] * data.shape[0])
    assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_1d_int(allclose):
    data = np.array([0.]*20+[0., 1., 2., 3., 4., 5., 3.]+[0.]*20)
    for npoints in range(1, 7):
        status = qc.islinear(data, npoints=npoints)

        expected = np.zeros(data.shape[0])
        if npoints<=4:
            expected[20:26] = 1

        assert allclose(status, expected)


@pytest.mark.skipif(not has_c_module("data", False), reason=SKIPMESS)
def test_islinear_sequence(allclose):
    nval = 50
    data = np.random.uniform(size=nval)
    ia1, ia2 = 20, 24
    data[ia1:ia2+1] = np.linspace(0, 1, 5)

    ib1, ib2 = 26, 31
    data[ib1:ib2+1] = np.linspace(0, 1, 6)

    for npoints in range(1, 10):
        status = qc.islinear(data, npoints)

        expected = np.zeros(nval)
        if npoints <= 3:
            expected[ia1:ia2+1] = 1

        if npoints <= 4:
            expected[ib1:ib2+1] = 1

        assert allclose(status, expected)

