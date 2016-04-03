import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.stat.distributions import LogNormCensored

class LogNormCensoredTestCase(unittest.TestCase):

    def setUp(self):
        print('\n\t=> LogNormCensoredTestCase (hystat)')
        FTEST = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FTEST

    def test_pdf(self):
        mu = 0.
        sig = 1.
        shift = 0.5
        lnc = LogNormCensored()
        lnc.pdf(2., mu, sig, shift)


    def test_fit(self):
        nval = 5000
        mu = 1.
        sig = 2.
        shift = -0.9
        censor = 1.
        x = np.exp(np.random.normal(mu, sig, size=nval))-shift
        x[x<censor] = censor

        lnc = LogNormCensored(censor)
        params = lnc.fit(x)
        ck = np.allclose(params[:3], (mu, sig, shift), rtol=0.05)

