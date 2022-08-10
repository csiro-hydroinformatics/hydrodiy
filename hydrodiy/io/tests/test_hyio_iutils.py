import os, re, json, sys
from pathlib import Path
import pytest

from string import ascii_letters as letters

import warnings

import pandas as pd
import subprocess
import numpy as np

import matplotlib.pyplot as plt

from hydrodiy.io import iutils, csv
from hydrodiy import PYVERSION

from requests.exceptions import HTTPError

SRC = Path(__file__).resolve()
FTEST = SRC.parent
FRUN = FTEST / "run_scripts"
FDATA = FRUN / "data"
FSCRIPTS = FRUN / "scripts"
for f in [FRUN, FDATA, FSCRIPTS]:
    f.mkdir(exist_ok=True)


def run_script(fs, stype="python"):
    """ Run script and check there are no errors in stderr """

    # Run system command
    pipe = subprocess.Popen(["python", fs, "-v 1"], \
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

    # Get outputs
    stdout, stderr = pipe.communicate()

    # detect errors
    hasError = False
    if len(stderr)>0:
        stderr = str(stderr)
        hasError = bool(re.search("Error", stderr))

        # Bypass install problems
        if re.search("No module named.*hydrodiy", stderr):
            hasError = False

    if hasError:
        print("STDERR not null in {0}:\n\t{1}".format(fs, stderr))

    # If no problem, then remove script
    if not hasError:
        os.remove(fs)

    return stderr, hasError


def test_script_template_default():
    sites = pd.DataFrame({"siteid":[1, 2, 3, 4], \
                "id":["a", "b", "c", "d"]})
    fs = FDATA / "sites.csv"
    csv.write_csv(sites, fs, "site list", SRC)

    # Run defaut script file template
    fs = FSCRIPTS / "script_test.py"
    comment = "This is a test script"
    iutils.script_template(fs, comment)

    assert fs.exists()
    with fs.open("r") as fo:
        txt = fo.read()
    assert re.search(f"## Comment : {comment}", txt)
    assert re.search(f"description=\\\"{comment}\\\"", txt)

    stderr, hasError = run_script(fs)
    if hasError:
        print(stderr)
    assert not hasError


def test_script_template_plot():
    sites = pd.DataFrame({"siteid":[1, 2, 3, 4], \
                "id":["a", "b", "c", "d"]})
    fs = FDATA / "sites.csv"
    csv.write_csv(sites, fs, "site list", SRC)

    # Run plot script file template
    fs = FSCRIPTS / "script_test.py"
    comment = "This is a plot test script"

    iutils.script_template(fs, comment, type="plot")

    assert fs.exists()
    with fs.open("r") as fo:
        txt = fo.read()
    assert re.search(f"## Comment : {comment}", txt)
    assert re.search(f"description=\\\"{comment}\\\"", txt)

    stderr, hasError = run_script(fs)
    assert not hasError


def test_str2dict():
    prefix = "this_is_a_prefix"

    data = {"name":"bob", "phone":"2010"}
    source = iutils.dict2str(data)
    data2, prefix2 = iutils.str2dict(source)
    source2 = iutils.dict2str(data2)
    assert prefix2 == ""
    assert data == data2
    assert source == source2

    source = FTEST / f"{source}.csv"
    data2, prefix2 = iutils.str2dict(source)
    source2 = iutils.dict2str(data2)
    assert prefix2 == ""
    assert data == data2
    assert re.sub("\\.csv", "", \
        os.path.basename(source)) == source2

    data = {"name":"bob", "phone":"2010"}
    source = iutils.dict2str(data, prefix=prefix)
    data2, prefix2 = iutils.str2dict(source)
    source2 = iutils.dict2str(data2, prefix2)
    assert data == data2
    assert prefix2 == prefix
    assert source == source2

    data = {"name":"bob_marley", "phone":"2010"}
    source = iutils.dict2str(data)
    data2, prefix2 = iutils.str2dict(source)
    source2 = iutils.dict2str(data2, prefix2)
    assert data == data2
    assert prefix2 == ""
    assert source == source2

    data = {"name":"bob_marley%$^_12234123", "phone":"2010"}
    source = iutils.dict2str(data)
    data2, prefix2 = iutils.str2dict(source)
    source2 = iutils.dict2str(data2, prefix2)
    assert data == data2

    data = {"name":"bob", "phone":2010}
    source = iutils.dict2str(data)
    data2, prefix2 = iutils.str2dict(source, False)
    source2 = iutils.dict2str(data2, prefix2)
    assert data == data2
    assert prefix2 == ""
    assert source == source2


def test_str2dict_random_order():
    """ Test order of arguments for str2dict """
    # Generate random keys
    nkeys = 10
    lkeys = 20

    l = [letters[k] for k in range(len(letters))]
    n = ["{0}".format(k) for k in range(10)]
    l = l+n

    d1 = {}
    for i in range(nkeys):
        key = "".join(np.random.choice(l, lkeys))
        value = "".join(np.random.choice(l, lkeys))
        d1[key] = value
    st1 = iutils.dict2str(d1)

    # Select random order of keys
    nrepeat = 100
    for i in range(nrepeat):
        d2 = {}
        keys = np.random.choice(list(d1.keys()), len(d1), \
                                replace=False)
        for key in keys:
            d2[key] = d1[key]

        # Generate string and compare with original one
        # This test checks that random perturbation of keys
        # does not affect the string
        st2 = iutils.dict2str(d2)
        assert st1==st2


def test_get_logger():
    """ Test Logger """
    flog1 = FTEST / "file1.log"
    flog2 = FTEST / "file2.log"

    # Test error on level
    try:
        logger = iutils.get_logger("bidule", level="INF")
    except ValueError as err:
        assert str(err).startswith("INF not a valid level")
    else:
        raise Exception("Problem with error handling")

    # Test logging
    logger1 = iutils.get_logger("bidule1", flog=flog1)

    mess = ["flog1 A", "flog1 B"]
    logger1.info(mess[0])
    logger1.info(mess[1])

    assert flog1.exists()

    with open(flog1, "r") as fl:
        txt = fl.readlines()
    ck = txt[0].strip().endswith("INFO | Process started")
    ck = ck & txt[1].strip().endswith("INFO | "+mess[0])
    ck = ck & txt[2].strip().endswith("INFO | "+mess[1])
    assert ck

    # Test logging with different format
    logger2 = iutils.get_logger("bidule2",
                    fmt="%(message)s",
                    flog=flog2)

    mess = ["flog2 A", "flog2 B"]
    logger2.warning(mess[0])
    logger2.critical(mess[1])

    assert flog2.exists()

    with open(flog2, "r") as fl:
        txt = fl.readlines()
    assert ["Process started"]+mess == [t.strip() for t in txt]

    # Close log file handler and delete files
    logger1.handlers[1].close()
    flog1.unlink()

    logger2.handlers[1].close()
    flog2.unlink()


def test_read_logfile():
    """ Test reading log data """
    flog = FTEST / "logfile.log"
    logs = iutils.read_logfile(flog)
    assert np.all(np.sort(logs.columns.tolist()) == \
            np.sort(["asctime", "levelname", "message", "name", "context"]))

    assert logs.shape, (70, 5)
    assert np.all(logs.context[:10] == "test context")
    assert np.all(logs.context[10:] == "")


def test_get_logger_contextual():
    """ Test contextual logger """
    flog = FTEST / "test_contextual.log"

    # Test logging
    logger = iutils.get_logger("bidule", flog=flog,\
                contextual=True)

    mess = ["flog1 A", "flog1 B"]
    logger.context = "context1"
    logger.info(mess[0])

    logger.context = "context2"
    logger.info(mess[1])

    assert flog.exists()

    with open(flog, "r") as fl:
        txt = fl.readlines()

    ck = bool(re.search("\\{ context1 \\}", txt[1]))
    ck = ck & bool(re.search("\\{ context2 \\}", txt[2]))
    assert ck

    logger.handlers[1].close()
    flog.unlink()


def test_thousands_separator():
    x = 56.1
    xfmt = iutils.thousands_separator(x)
    assert xfmt == "56.1"

    x = 1056.3
    xfmt = iutils.thousands_separator(x)
    assert xfmt == "1,056.3"

    x = 5101056.3
    xfmt = iutils.thousands_separator(x)
    assert xfmt == "5,101,056.3"

    x = 1056.3456
    xfmt = iutils.thousands_separator(x, dec=2)
    assert xfmt == "1,056.35"


