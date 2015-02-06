import os
import unittest
import numpy as np
from hywafari import wplots

import matplotlib.pyplot as plt

import pandas as pd

class WreviewTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> WreviewTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)
        self.FTEST = FTEST

    def test_wreview(self):

        url = 'http://www.bom.gov.au/water/ssf'
        log = '%s/wreview.log' % self.FTEST

        cmd = '%s/../wreview.py -u %s -f %s -t' % (self.FTEST, url ,log)
        os.system(cmd)

if __name__ == "__main__":
    unittest.main()



