import os
import unittest
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta as delta
import pandas as pd

from hydrodiy import HAS_C_DATA_MODULE

# Fix seed
np.random.seed(42)

class DateutilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> DateutilsTestCase (hydata)')

        self.months = pd.date_range('1800-01-01', '2200-12-1', freq='MS')
        self.days = pd.date_range('1800-01-01', '2200-12-1', freq='5D')

        if not HAS_C_DATA_MODULE:
            self.skipTest('Missing C modules')

    def test_isleapyear(self):
        years = range(1800, 2200)
        isleap = pd.Series(years).apply(lambda x:
                            '{0}-02-29'.format(x))
        isleap = pd.to_datetime(isleap, errors='coerce')
        isleap = pd.notnull(isleap)

        si = 0
        for y, i in zip(years, isleap):
            si += abs(int(i)-chd.isleapyear(y))
        self.assertTrue(si == 0)


    def test_daysinmonth(self):
        sn = 0
        for m in self.months:
            nb = ((m+delta(months=1) - delta(days=1))-m).days + 1
            sn += abs(nb-chd.daysinmonth(m.year, m.month))
        self.assertTrue(sn == 0)


    def test_dayofyear(self):
        sd = 0
        for d in self.days:
            nb = d.dayofyear

            # Correct for 29 Feb
            if chd.isleapyear(d.year) and d.month>=3:
                nb -= 1

            sd += abs(nb-chd.dayofyear(d.month, d.day))

        self.assertTrue(sd == 0)


    def test_add1month(self):
        sd = 0
        for d in self.days:
            d2 = d + delta(months=1)
            dd2 = np.array([d2.year, d2.month, d2.day])

            dd = np.array([d.year, d.month, d.day]).astype(np.int32)
            chd.add1month(dd)

            err = abs(np.sum(dd-dd2))
            sd += err

        self.assertTrue(sd == 0)


    def test_add1day(self):
        sd = 0
        for d in self.days:
            d2 = d + delta(days=1)
            dd2 = np.array([d2.year, d2.month, d2.day])

            dd = np.array([d.year, d.month, d.day]).astype(np.int32)
            chd.add1day(dd)

            err = abs(np.sum(dd-dd2))
            sd += err

        self.assertTrue(sd == 0)


    def test_comparedates(self):
        ntrial = 5000
        sd = 0
        for i in range(ntrial):
            k1 = np.random.choice(range(len(self.days)), 1)
            d1 = self.days[k1]
            dd1 = np.array([d1.year, d1.month, d1.day]).astype(np.int32)

            k2 = np.random.choice(range(len(self.days)), 1)
            d2 = self.days[k2]
            dd2 = np.array([d2.year, d2.month, d2.day]).astype(np.int32)

            diffa = 0
            if d1<d2:
                diffa = 1
            if d1>d2:
                diffa = -1
            diffb = chd.comparedates(dd1[:, 0], dd2[:, 0])
            err = abs(diffa-diffb)
            sd += err

        self.assertTrue(sd == 0)


    def test_getdate(self):
        sd = 0
        for d in self.days:
            day = d.year*1e4 + d.month*1e2 + d.day
            dt = np.array([d.year, d.month, d.day]).astype(np.int32)
            dt2 = dt*0
            chd.getdate(day, dt2)

            err = abs(np.sum(dt-dt2))
            sd += err

        self.assertTrue(sd == 0)



if __name__ == "__main__":
    unittest.main()
