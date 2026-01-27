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
    #if not hasError:
    #    fs.unlink()

    return stderr, hasError


@pytest.mark.parametrize("stype", iutils.SCRIPT_TYPES)
def test_script_template(stype):
    sites = pd.DataFrame({"stationid":[1, 2, 3, 4], \
                "id":["a", "b", "c", "d"]})
    fs = FDATA / "stations.csv"
    csv.write_csv(sites, fs, "site list", SRC)

    # Run defaut script file template
    fs = FSCRIPTS / f"script_template_{stype}.py"
    comment = "This is a test script"
    iutils.script_template(fs, comment, stype)

    assert fs.exists()
    with fs.open("r") as fo:
        txt = fo.read()
    assert re.search(f"## Comment : {comment}", txt)

    if stype != "default":
        assert re.search(f"description=\\\"{comment}\\\"", txt)

    stderr, hasError = run_script(fs)
    if hasError:
        print(stderr)
    assert not hasError


def test_script_template_plot():
    sites = pd.DataFrame({"siteid":[1, 2, 3, 4], \
                "id":["a", "b", "c", "d"]})
    fs = FDATA / "stations.csv"
    csv.write_csv(sites, fs, "site list", SRC)

    # Run plot script file template
    fs = FSCRIPTS / "script_template_plot_add.py"
    comment = "This is a plot test script"

    iutils.script_template(fs, comment, stype="plot")

    assert fs.exists()
    with fs.open("r") as fo:
        txt = fo.read()
    assert re.search(f"## Comment : {comment}", txt)
    assert re.search(f"description=\\\"{comment}\\\"", txt)

    stderr, hasError = run_script(fs)
    assert not hasError


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
    logger1.info("test tab1", 1)
    logger1.info("test tab2", 2)
    logger1.error("error")
    logger1.critical("critical")
    logger1.warning("warning")
    logger1.log_dict({"a": 1, "b": "asdasd"}, "test", "warning")

    assert flog1.exists()

    with open(flog1, "r") as fl:
        txt = fl.readlines()

    ck = txt[0].strip().endswith("INFO | @@@ Process started @@@")
    ck = ck & txt[3].strip().endswith("INFO | "+mess[0])
    ck = ck & txt[4].strip().endswith("INFO | "+mess[1])
    ck = ck & txt[5].strip().endswith("INFO |     test tab1")
    ck = ck & txt[6].strip().endswith("INFO |         test tab2")
    ck = ck & txt[7].strip().endswith("ERROR | error")
    ck = ck & txt[8].strip().endswith("CRITICAL | critical")
    ck = ck & txt[9].strip().endswith("WARNING | warning")
    ck = ck & txt[11].strip().endswith("WARNING | test:")
    ck = ck & txt[12].strip().endswith("WARNING |     a = 1")

    assert ck

    # Test logging with different format
    logger2 = iutils.get_logger("bidule2",
                    fmt="%(message)s",
                    flog=flog2)

    mess = ["flog2 A", "flog2 B"]
    logger2.warning(mess[0], nret=2)
    logger2.critical(mess[1], ntab=2)
    logger2.completed()

    assert flog2.exists()

    with open(flog2, "r") as fl:
        txt = fl.readlines()
    expected = ["@@@ Process started @@@",
                logger2.separator_charac*logger2.separator_length,
                "", "", ""] +\
               mess +\
               ["",
                logger2.separator_charac*logger2.separator_length,
                "Execution time : 0h 0m 0s",
                "@@@ Process completed @@@"]
    assert expected == [t.strip() for t in txt]

    # Close log file handler and delete files
    logger1.handlers[1].close()
    flog1.unlink()

    logger2.handlers[1].close()
    flog2.unlink()


def test_get_logger_contextual():
    """ Test contextual logger """
    flog = FTEST / "test_contextual.log"

    # Test logging
    logger = iutils.get_logger("bidule", flog=flog,\
                contextual=True)
    logger.context_hasheader = True

    mess = ["flog1 A", "flog1 B"]
    logger.context = "context1"
    logger.info(mess[0])

    logger.context = "context2"
    logger.info(mess[1])
    logger.error("error")
    logger.critical("critical")
    logger.warning("warning")
    logger.info("test tab3", 3)

    logger.completed()

    assert flog.exists()

    with open(flog, "r") as fl:
        txt = fl.readlines()

    ck = bool(re.search("@@@ Process started @@@", txt[0]))
    ck &= bool(re.search("\\{ context1 \\}", txt[4]))
    ck &= bool(re.search("\\{ context2 \\}", txt[7]))
    ck &= bool(re.search("ERROR . \\{ context2 \\}", txt[9]))
    ck &= bool(re.search("CRITICAL . \\{ context2 \\}", txt[10]))
    ck &= bool(re.search("WARNING . \\{ context2 \\}", txt[11]))
    ck &= bool(re.search("INFO . \\{ context2 \\}             test tab3", txt[12]))
    ck &= bool(re.search("Execution time", txt[16]))
    ck &= bool(re.search("@@@ Process completed @@@", txt[17]))
    assert ck

    logger.handlers[1].close()
    flog.unlink()


