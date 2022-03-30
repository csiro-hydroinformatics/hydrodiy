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


def test_sitebatch():
    siteids = ["a", "b", "c", "d", "e", "f"]
    sb = hyruns.SiteBatch(siteids, 2)
    assert sb.siteids.tolist() == siteids
    b = sb[1]
    assert b == ["d", "e", "f"]

    ib = sb.search("d")
    assert ib == 1

    msg = "Non unique"
    with pytest.raises(AssertionError, match=msg):
        sb = hyruns.SiteBatch(["a", "a", "b"], 2)



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


def test_task_print():
    t = hyruns.OptionTask(0, {"bidule": "truc"}, \
                    {"opt1": 1, "opt2": 3})
    s = str(t)


def test_task_dict():
    t = hyruns.OptionTask(0, {"bidule": "truc"}, \
                    {"opt1": 1, "opt2": 3})
    dd = t.to_dict()
    assert dd == {"taskid": 0, \
                    "context": {"bidule": "truc"},
                    "options": {"opt1": 1, "opt2": 3}
                 }

    t2 = hyruns.OptionTask.from_dict(dd)
    assert str(t) == str(t2)


def test_option_manager():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    assert opm.ntasks == 6
    assert opm.bidule == "test"

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

    fout = TESTS_DIR / "option_manager.json"
    opm.save(fout)
    assert fout.exists()

    with fout.open("r") as fo:
        js = json.load(fo)
    assert not set(list(js.keys())) ^ set(["name", "tasks", "context", "options"])

    fout.unlink()


def test_option_manager_changed_keys():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    hyruns.set_dict_keyname("options", "items")
    fout = TESTS_DIR / "option_manager.json"
    opm.save(fout, overwrite=True)
    with fout.open("r") as fo:
        js = json.load(fo)
    assert not set(list(js.keys())) ^ set(["name", "tasks", "context", "items"])

    hyruns.set_dict_keyname("options", "truc")
    opm.save(fout, overwrite=True)
    with fout.open("r") as fo:
        js = json.load(fo)
    assert not set(list(js.keys())) ^ set(["name", "tasks", "context", "truc"])

    msg = "Expected key in"
    with pytest.raises(AssertionError, match=msg):
        hyruns.set_dict_keyname("truc", "truc")

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

    found = opm.search(v1="a", v2=1)
    assert found == [0, 1]

    msg = "Expected option 'bidule' in"
    with pytest.raises(AssertionError, match=msg):
        found = opm.search(bidule="a")


def test_option_manager_find():
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(v1=["a", "b"], \
                    v2=[1, 2, 3], v3=[[1, 2], [3, 4]])

    found = opm.find(v3=[1, 2])
    assert found == [0, 2, 4, 6, 8, 10]

    found = opm.find(v1="a", v2=1)
    assert found == [0, 1]

    msg = "Expected option 'bidule' in"
    with pytest.raises(AssertionError, match=msg):
        found = opm.search(bidule="a")


def test_option_manager_match():
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(v1=["a", "b"], \
                    v2=[1, 2, 3], v3=[[1, 2], [3, 4]])

    task = hyruns.OptionTask(0, {}, {"v1": "a", "v2": 1, "v3":[1, 2]})
    taskids = opm.match(task)
    assert taskids == [0]

    task = hyruns.OptionTask(0, {}, {"v1": "c", "v2": 1, "v3":[1, 2]})
    taskids = opm.match(task)
    assert taskids == []


    task = hyruns.OptionTask(0, {}, {"v1": "a", "v2": 1, "v4": 1})
    msg = "Expected option 'v4' in"
    with pytest.raises(AssertionError, match=msg):
        taskids = opm.match(task)

    task = hyruns.OptionTask(0, {}, {"v1": "a", "v2": 1, "v4": 1})
    taskids = opm.match(task, exclude="v4")
    assert taskids == [0, 1]

    # Match with a sub selection
    taskids = opm.match(task, exclude="v4", v3=[1, 2])
    assert taskids == [0]

    taskids = opm.match(task, exclude=["v2", "v4"], v2=1)
    assert taskids == [0, 1]


def test_option_manager_single_values():
    opm = hyruns.OptionManager()
    opm.from_cartesian_product(v1="a", v2=[1, 2, 3])

    assert opm.ntasks == 3
    t = opm.get_task(2)
    assert t.names == ["v1", "v2"]
    assert t.v1 == "a"
    assert t.v2 == 3


def test_option_manager_equality():
    opm1 = hyruns.OptionManager(bidule=[1, 2])
    opm1.from_cartesian_product(v1="a", v2=[1, 2, 3])

    opm2 = hyruns.OptionManager(bidule=[1, 2])
    opm2.from_cartesian_product(v1="a", v2=[1, 2, 3])
    assert opm1 == opm2

    opm3 = hyruns.OptionManager()
    opm3.from_cartesian_product(v1="a", v2=[1, 2, 3])
    assert opm1 != opm3

    opm4 = hyruns.OptionManager(bidule=[1, 2])
    opm4.from_cartesian_product(v1="a", v2=[1, 2])
    assert opm1 != opm4

    assert "bidule" != opm1
    assert opm1 != "bidule"


def test_option_dict():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    dd = opm.to_dict()
    opm2 = hyruns.OptionManager.from_dict(dd)
    assert opm.context == opm2.context
    assert opm.name == opm.name
    for t1, t2 in zip(opm.tasks, opm2.tasks):
        assert t1 == t2

    assert str(opm) == str(opm2)


def test_option_file():
    opm = hyruns.OptionManager(bidule="test")
    opm.from_cartesian_product(v1=["a", "b"], v2=[1, 2, 3])

    f = TESTS_DIR / "opm.json"
    if f.exists():
        f.unlink()

    opm.save(f)
    t1 = f.stat().st_mtime

    opm.save(f)
    t2 = f.stat().st_mtime
    # Check save has not overwritten an existing file
    assert t1 == t2

    opm2 = hyruns.OptionManager.from_file(f)
    assert opm.context == opm2.context

    f.unlink()

