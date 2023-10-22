import os
import re
import pytest
import math

import json

from itertools import product

from timeit import Timer
import time

import pandas as pd
import numpy as np
np.seterr(all="print")

from scipy.special import kolmogorov

from hydrodiy.data.containers import Vector

FHERE = os.path.dirname(os.path.abspath(__file__))

def test_init():
    # Basic instances
    vect = Vector("a")

    vect = Vector(["a", "b"])

    msg = "Names are not"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "a"])

    msg = "Expected vector of length"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"], defaults=1)

    msg = "Expected vector of length"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"], defaults=[1]*2, mins=1)

    msg = "Expected vector of length"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"], defaults=[1]*2, mins=[0]*2, maxs=2)

    msg = "Expected maxs within"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"], defaults=[1]*2, mins=[0]*2, maxs=[-1, 0.5])

    msg = "Expected defaults within"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"], defaults=[1]*2, mins=[0]*2, \
                                    maxs=[0.5]*2)


def test_empty_vector():
    for vect in [Vector([]), Vector(None)]:
        assert vect.nval==0
        msg = "Expected vector of length"
        with pytest.raises(ValueError, match=msg):
            vect.values = 0

        dct = vect.to_dict()
        assert dct ==  {"check_bounds":True, \
                                "check_hitbounds":False, \
                                "accept_nan":False, \
                                    "hitbounds":False, \
                                    "data":[], "nval":0}


def test_tofromdict():
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2)
    dct = vect.to_dict()
    vect2 = Vector.from_dict(dct)
    dct2 = vect2.to_dict()
    assert dct==dct2


def test_string():
    vect = Vector(["a", "b"])
    vect.values = [1. ,1.]
    print(vect)


def test_set_get():
    vect = Vector(["a", "b", "c"], [0.5]*3, [0]*3, [1]*3)
    values = np.linspace(0, 1, 3)
    expected = np.zeros(3)
    for i, nm in enumerate(vect.names):
        vect[nm] = values[i]
        expected[i] = vect[nm]

    assert np.allclose(values, expected)

    vect["a"] = 10
    assert np.allclose(vect["a"], 1.)

    vect["a"] = -10
    assert np.allclose(vect["a"], 0.)

    msg = "Cannot set value to nan"
    with pytest.raises(ValueError, match=msg):
        vect["a"] = np.nan


def test_set_get_attributes():
    vect = Vector(["a", "b", "c"], [0.5]*3, [0]*3, [1]*3)

    vect.a = 0.8
    assert np.allclose(vect.values, [0.8, 0.5, 0.5])
    assert np.allclose(vect["a"], 0.8)

    vect.a = 2.
    assert np.allclose(vect.values, [1., 0.5, 0.5])
    assert np.allclose(vect["a"], 1.)

    vect.a = -2.
    assert np.allclose(vect.values, [0., 0.5, 0.5])
    assert np.allclose(vect["a"], 0.)


def test_hitbounds():
    # Test vector with active hitbounds checking
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2, \
                check_hitbounds=True)
    assert ~vect.hitbounds

    vect.values = [2]*2
    assert vect.hitbounds

    vect.reset()
    assert ~vect.hitbounds

    vect.values = [-2]*2
    assert vect.hitbounds

    # Test vector with not hitbound checking
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2, False)
    assert ~vect.hitbounds

    vect.values = [2]*2
    assert ~vect.hitbounds


def test_values():
    msg = "Expected vector of length"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"])
        vect.values = 1

    msg = "Cannot process value"
    with pytest.raises(ValueError, match=msg):
        vect = Vector(["a", "b"])
        vect.values = [1., np.nan]

    vect = Vector(range(4))
    vect.values = np.arange(4).reshape((2, 2))
    assert len(vect.values.shape) == 1

    msg = "could not convert"
    with pytest.raises(ValueError, match=msg):
        vect.values = ["a"]*4


def test_clone():
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2)
    vect.values = [0.6]*2
    dct = vect.to_dict()

    vect2 = vect.clone()
    dct2 = vect.to_dict()
    assert dct==dct2

    vect2["a"] = 1.


def test_reset():
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2)

    vect.values = [0.7]*2
    vect.reset()
    assert np.allclose(vect.values, vect.defaults)


def test_accept_nan():
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2, \
                accept_nan = True)
    vect.a = np.nan


def test_to_series():
    vect = Vector(["a", "b"], [0.5]*2, [0]*2, [1]*2)
    se = vect.to_series()
    assert np.all([idx1 == idx2 for idx1, idx2 in zip(se.index, ["a", "b"])])


