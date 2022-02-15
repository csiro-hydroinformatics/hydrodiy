import os, re, json, sys
from pathlib import Path

import pytest

import warnings

import pandas as pd
import numpy as np

from hydrodiy.io import hyruns

filename = Path(__file__).resolve()
TESTS_DIR = filename.parent

def test_get_batch():
    """ Test get_batch for small batches """
    nbatch = 5
    nsites = 26
    idx = [hyruns.get_batch(nsites, nbatch, ibatch) \
                for ibatch in range(nbatch)]
    prev = -1
    for i, ii in enumerate(idx):
        assert np.all(np.diff(ii) == 1)

        assert ii[0] == prev+1
        prev = ii[-1]

        assert len(ii) == (6 if i==0 else 5)

    assert idx[-1][0] == 21
    assert idx[-1][1] == 22


def test_get_batch_errors():
    nbatch = 5
    nsites = 26
    idx = [hyruns.get_batch(nsites, nbatch, ibatch) \
                for ibatch in range(nbatch)]

    msg = "Expected nelements"
    with pytest.raises(ValueError, match=msg):
        idx = hyruns.get_batch(20, 40, 1)

    msg = "Expected ibatch"
    with pytest.raises(ValueError, match=msg):
        idx = hyruns.get_batch(40, 5, 7)


def test_get_batch_large():
    """ Test get_batch for large batches """
    nbatch = 6
    nsites = 502
    idx = [hyruns.get_batch(nsites, nbatch, ibatch) \
                for ibatch in range(nbatch)]

    prev = -1
    for i, ii in enumerate(idx):
        assert np.all(np.diff(ii) == 1)

        assert ii[0] == prev+1
        prev = ii[-1]

        assert len(ii) == (84 if i<nbatch-2 else 83)


def test_option_manager():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    assert opm.ntasks == 6

    t = opm.get_task(0)
    print(t)
    assert t.names == ["v1", "v2"]
    assert t.context["bidule"] == "test"
    assert t.v1 == "a"
    assert t.bidule == "test"
    assert t["bidule"] == "test"
    assert t.v1 == "a"
    assert t["v1"] == "a"
    assert t.v2 == 1
    assert t["v2"] == 1

    t = opm.get_task(1)
    assert t.names == ["v1", "v2"]
    assert t.context["bidule"] == "test"
    assert t.v1 == "a"
    assert t.bidule == "test"
    assert t["bidule"] == "test"
    assert t.v1 == "a"
    assert t.v2 == 2

    msg = "Expected taskid in"
    with pytest.raises(AssertionError, match=msg):
        t = opm.get_task(10)

    print(opm)

    df = opm.to_dataframe()
    assert df.shape == (6, 2)
    assert df.columns.tolist() == ["v1", "v2"]

    fout = TESTS_DIR / "option_manager.csv"
    opm.save(fout)
    assert fout.exists()
    fout.unlink()

class FakeLogger():
    def __init__(self):
        self.content = []

    def info(self, line):
        self.content.append(line)


def test_task_log():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    t = opm.get_task(0)
    logger = FakeLogger()
    t.log(logger)

    content = ['', '****** TASK 0 *******', \
                'Context bidule: test', '', \
                'Item v1: a', 'Item v2: 1', \
                '***********************', '']
    assert logger.content == content


def test_option_manager_search():
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(v1=["a", "b"], \
                    v2=[1, 2, 3], v3=[[1, 2], [3, 4]])
    found = opm.search(v1="a", v2="1|3")
    assert found == [0, 1, 4, 5]

    msg = "Expected option 'bidule' in"
    with pytest.raises(AssertionError, match=msg):
        found = opm.search(bidule="a")


def test_option_manager_single_values():
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(v1="a", v2=[1, 2, 3])

    assert opm.ntasks == 3
    t = opm.get_task(2)
    assert t.names == ["v1", "v2"]
    assert t.v1 == "a"
    assert t.v2 == 3


