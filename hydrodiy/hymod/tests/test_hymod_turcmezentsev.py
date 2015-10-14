import os
import re
import unittest

from timeit import Timer
import time

import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.models.turcmezentsev import TurcMezentsev

class TurcMezentsevTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> TurcMezentsevTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
    
    def test_tm(self):
        P = np.linspace(700, 1200, 20)
        PE = np.linspace(1000, 2000, 20)
        inputs = np.concatenate([P[:,None], PE[:, None]], axis=1)

        tm = TurcMezentsev()
        n = 2.3
        tm.set_trueparams(n)
        tm.run(inputs)
        Q1 = tm.outputs
        Q2 = P*(1-1/(1+(P/PE)**n)**(1/n))
        self.assertTrue(all(Q1==Q2))


if __name__ == "__main__":
    unittest.main()
