import os
import unittest
import numpy as np
from hywafari import wdata

import pandas as pd

class DataTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> DataTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)
        self.FTEST = FTEST

    def test_get_sites(self):
        sites = wdata.get_sites()
        self.assertTrue(isinstance(sites, pd.DataFrame))

    def test_get_monthly(self):
        qobs1 = wdata.get_monthlyflow('410734')
        self.assertTrue(isinstance(qobs1, pd.Series))

        qobs2 = wdata.get_monthlyflow('bidule')
        self.assertTrue(qobs2 is None)

    def test_get_daily(self):
        d1 = wdata.get_daily('410734')
        self.assertTrue(isinstance(d1, pd.DataFrame))

        d2 = wdata.get_daily('bidule')
        self.assertTrue(d2 is None)

if __name__ == "__main__":
    unittest.main()

