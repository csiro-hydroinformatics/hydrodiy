import os, re, json, sys
from pathlib import Path
import unittest

from string import ascii_letters as letters

import warnings

import pandas as pd
import subprocess
import numpy as np

import matplotlib.pyplot as plt

from hydrodiy.io import iutils, csv
from hydrodiy import PYVERSION

from requests.exceptions import HTTPError

# Skip if package cannot be imported (circleci build)
import_error = True
try:
    from hydrodiy.gis.oz import Oz, REGIONS
    import_error = False
except ImportError:
    pass


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


class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print("\t=> UtilsTestCase (hyio)")
        self.source_file = Path(__file__).resolve()
        self.ftest = self.source_file.parent

        self.frun = self.ftest / "run_scripts"
        self.fdata = self.frun / "data"
        self.fscripts = self.frun / "scripts"
        for f in [self.frun, self.fdata, self.fscripts]:
            f.mkdir(exist_ok=True)


    def test_script_template_default(self):
        """ Test running script template default """
        sites = pd.DataFrame({"siteid":[1, 2, 3, 4], \
                    "id":["a", "b", "c", "d"]})
        fs = self.fdata / "sites.csv"
        csv.write_csv(sites, fs, "site list", self.source_file)

        # Run defaut script file template
        fs = self.fscripts / "script_test.py"
        iutils.script_template(fs, "test")

        stderr, hasError = run_script(fs)
        self.assertFalse(hasError)


    def test_script_template_plot(self):
        """ Test running script template plot """
        if import_error:
            self.skipTest("Import error")

        sites = pd.DataFrame({"siteid":[1, 2, 3, 4], \
                    "id":["a", "b", "c", "d"]})
        fs = self.fdata / "sites.csv"
        csv.write_csv(sites, fs, "site list", self.source_file)

        # Run plot script file template
        fs = self.fscripts / "script_test.py"
        iutils.script_template(fs, "test", type="plot")
        stderr, hasError = run_script(fs)
        self.assertFalse(hasError)


    def test_str2dict(self):
        """ Test str2dict """
        prefix = "this_is_a_prefix"

        data = {"name":"bob", "phone":"2010"}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2)
        self.assertTrue(prefix2 == "")
        self.assertTrue(data == data2)
        self.assertTrue(source == source2)

        source = os.path.join(self.ftest, source + ".csv")
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2)
        self.assertTrue(prefix2 == "")
        self.assertTrue(data == data2)
        self.assertTrue(re.sub("\\.csv", "", \
            os.path.basename(source)) == source2)

        data = {"name":"bob", "phone":"2010"}
        source = iutils.dict2str(data, prefix=prefix)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == prefix)
        self.assertTrue(source == source2)

        data = {"name":"bob_marley", "phone":"2010"}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == "")
        self.assertTrue(source == source2)

        data = {"name":"bob_marley%$^_12234123", "phone":"2010"}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        #self.assertTrue(prefix2 == "")
        #self.assertTrue(source == source2)

        data = {"name":"bob", "phone":2010}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source, False)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == "")
        self.assertTrue(source == source2)


    def test_str2dict_random_order(self):
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
            self.assertTrue(st1==st2)



    def test_get_logger(self):
        """ Test Logger """
        flog1 = os.path.abspath(__file__) + "1.log"
        flog2 = os.path.abspath(__file__) + "2.log"

        # Test error on level
        try:
            logger = iutils.get_logger("bidule", level="INF")
        except ValueError as err:
            self.assertTrue(str(err).startswith("INF not a valid level"))
        else:
            raise Exception("Problem with error handling")

        # Test logging
        logger1 = iutils.get_logger("bidule1", flog=flog1)

        mess = ["flog1 A", "flog1 B"]
        logger1.info(mess[0])
        logger1.info(mess[1])

        self.assertTrue(os.path.exists(flog1))

        with open(flog1, "r") as fl:
            txt = fl.readlines()
        ck = txt[0].strip().endswith("INFO | "+mess[0])
        ck = ck & txt[1].strip().endswith("INFO | "+mess[1])
        self.assertTrue(ck)

        # Test logging with different format
        logger2 = iutils.get_logger("bidule2",
                        fmt="%(message)s",
                        flog=flog2)

        mess = ["flog2 A", "flog2 B"]
        logger2.warn(mess[0])
        logger2.critical(mess[1])

        self.assertTrue(os.path.exists(flog2))

        with open(flog2, "r") as fl:
            txt = fl.readlines()
        self.assertEqual(mess, [t.strip() for t in txt])

        # Close log file handler and delete files
        logger1.handlers[1].close()
        os.remove(flog1)

        logger2.handlers[1].close()
        os.remove(flog2)


    def test_read_logfile(self):
        """ Test reading log data """
        flog = os.path.join(self.ftest, "logfile.log")
        logs = iutils.read_logfile(flog)
        self.assertCountEqual(list(logs.columns), \
                ["asctime", "levelname", "message", "name", "context"])

        self.assertEqual(logs.shape, (70, 5))
        self.assertTrue(np.all(logs.context[:10] == "test context"))
        self.assertTrue(np.all(logs.context[10:] == ""))


    def test_get_logger_contextual(self):
        """ Test contextual logger """
        flog = os.path.abspath(__file__) + ".log"

        # Test logging
        logger = iutils.get_logger("bidule", flog=flog,\
                    contextual=True)

        mess = ["flog1 A", "flog1 B"]
        logger.context = "context1"
        logger.info(mess[0])

        logger.context = "context2"
        logger.info(mess[1])

        self.assertTrue(os.path.exists(flog))

        with open(flog, "r") as fl:
            txt = fl.readlines()

        ck = bool(re.search("\{ context1 \}", txt[0]))
        ck = ck & bool(re.search("\{ context2 \}", txt[1]))
        self.assertTrue(ck)


    def test_get_ibatch(self):
        """ Test get_ibatch for small batches """
        nbatch = 5
        nsites = 26
        idx = [iutils.get_ibatch(nsites, nbatch, ibatch) \
                    for ibatch in range(nbatch)]
        prev = -1
        for i, ii in enumerate(idx):
            self.assertTrue(np.all(np.diff(ii) == 1))

            self.assertTrue(ii[0] == prev+1)
            prev = ii[-1]

            self.assertTrue(len(ii) == (6 if i==0 else 5))

        self.assertTrue(idx[-1][0] == 21)
        self.assertTrue(idx[-1][1] == 22)

        try:
            idx = iutils.get_ibatch(20, 40, 1)
        except ValueError as err:
            self.assertTrue(str(err).startswith(\
                            "Expected nsites"))
        else:
            raise Exception("Problem with error handling")

        try:
            idx = iutils.get_ibatch(40, 5, 7)
        except ValueError as err:
            self.assertTrue(str(err).startswith(\
                    "Expected ibatch"))
        else:
            raise Exception("Problem with error handling")


    def test_get_ibatch_large(self):
        """ Test get_ibatch for large batches """
        nbatch = 6
        nsites = 502
        idx = [iutils.get_ibatch(nsites, nbatch, ibatch) \
                    for ibatch in range(nbatch)]

        prev = -1
        for i, ii in enumerate(idx):
            self.assertTrue(np.all(np.diff(ii) == 1))

            self.assertTrue(ii[0] == prev+1)
            prev = ii[-1]

            self.assertTrue(len(ii) == (84 if i<nbatch-2 else 83))


if __name__ == "__main__":
    unittest.main()
