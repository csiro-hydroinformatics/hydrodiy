import os
import unittest
import numpy as np
import datetime
import pandas as pd
from hydata import dutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hydata)')
        self.dt = None # TODO

    def test_wyear1(self):
        day = datetime.datetime(2001, 12, 3)
        yw = dutils.wyear(day)
        self.assertEqual(yw, 2001)

    def test_wyear2(self):
        day = datetime.datetime(2001, 1, 3)
        yw = dutils.wyear(day)
        self.assertEqual(yw, 2000)

    def test_wyear_days1(self):
        day = datetime.datetime(2000, 2, 10)
        wday = dutils.wyear_days(day, start_month=2)
        self.assertEqual(wday, 10)

    def test_wyear_days2(self):
        day = datetime.datetime(2000, 12, 31)
        wday = dutils.wyear_days(day, start_month=1)
        self.assertEqual(wday, 366)

    def test_cycledist(self):
        self.assertEqual(dutils.cycledist(1, 2), 1)
        self.assertEqual(dutils.cycledist(1, 11), 2)
        self.assertEqual(dutils.cycledist(1, 7), 6)
        self.assertEqual(dutils.cycledist(1, 5, start=1, end=5), 1)
        self.assertEqual(dutils.cycledist(1, 4, start=1, end=5), 2)
        self.assertEqual(dutils.cycledist(1, 365, start=1, end=365), 1)
        self.assertEqual(dutils.cycledist(1, 360, start=1, end=365), 6)

    def test_runclim(self):
        index = pd.date_range('1900-01-01', '2020-12-31', freq='d')
        n = len(index)
        u = np.sin((0.+index.dayofyear)/366*2*np.pi)
        u += 0.2*np.random.uniform(size=n)-0.1
        s = pd.Series(u, index=index)
        clim = dutils.runclim(s)
        v = np.sin((0.+np.arange(366))/366*2*np.pi)
        err = np.abs(clim['median']-v)/(1+np.abs(v))
        self.assertTrue(np.max(err)<0.03)

    def test_runclimcum(self):
        index = pd.date_range('1950-01-01', '2020-12-31', freq='D')
        n = len(index)
        u = np.sin((0.+index.dayofyear)/366*2*np.pi)
        u += 0.5*np.random.uniform(size=n)-0.25
        u = u*u
        s = pd.Series(u, index=index)
        climc = dutils.runclimcum(s)

        #import matplotlib.pyplot as plt
        #climc[['lowest', 'median', 'highest']].plot(); plt.show()
        #TODO

if __name__ == "__main__":
    unittest.main()
