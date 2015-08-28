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

    def test_get_monthly(self):
        qobs = wdata.get_monthly('410734')

    def test_get_daily(self):
        data = wdata.get_daily('410734')

if __name__ == "__main__":
    unittest.main()

