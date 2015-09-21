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

    def test_normaliseid(self):

        id = 'GGSDFFSFsdfsdf'
        idn = dutils.normaliseid(id)
        self.assertEqual(idn, id.upper())

        id = '0000101'
        idn = dutils.normaliseid(id)
        self.assertEqual(idn, '101')

        id = '105GA'
        idn = dutils.normaliseid(id)
        self.assertEqual(idn, '105')

        id = '105GA - asd 10'
        idn = dutils.normaliseid(id)
        self.assertEqual(idn, '105GA_ASD10')

        id = '101.101'
        idn = dutils.normaliseid(id)
        self.assertEqual(idn, '101')

    def test_ordinalsec(self):

        def rn(x0, x1):
            return np.random.randint(x0, x1)

        n = 5000

        for i in range(n):
            t = datetime.datetime(rn(500, 9900), rn(1, 12), rn(1, 28),
                rn(0, 23), rn(0, 59), rn(0, 59))

            o = dutils.time2osec(t)

            self.assertTrue(isinstance(o, np.uint64))

            self.assertEqual(t.toordinal(), int(o/86400))

            s = int(o - np.uint64(t.toordinal())*86400)

            self.assertEqual(t.hour, s/3600)

            t2 = dutils.osec2time(o)

            self.assertEqual(t, t2)

    def test_secofyear(self):

        t = datetime.datetime(2001, 1, 1)
        n = dutils.secofyear(t)

        self.assertTrue(isinstance(n, np.uint64))
        self.assertEqual(n, 0)

        t1 = datetime.datetime(2000, 2, 29)
        n1 = dutils.secofyear(t1)

        t2 = datetime.datetime(2000, 3, 1)
        n2 = dutils.secofyear(t2)

        self.assertTrue(isinstance(n1, np.uint64))
        self.assertTrue(isinstance(n2, np.uint64))
        self.assertEqual(n1, n2)

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
        u0 = 5 + np.sin((0.+index.dayofyear)/365*2*np.pi)
        u  = u0 + 0.5*(np.random.uniform(size=n)-0.5)
        s = pd.Series(u, index=index)
        clim, yws = dutils.runclim(s, nwin=30)
        v = 5 + np.sin((0.+np.arange(365))/365*2*np.pi)
        err = np.abs(clim['50%']-v)/(1+np.abs(v))
        self.assertTrue(np.max(err)<0.03)

    def test_runclimcum(self):
        index = pd.date_range('1950-01-01', '2020-12-31', freq='D')
        n = len(index)
        u = np.sin((0.+index.dayofyear)/366*2*np.pi)
        u += 1.*(np.random.uniform(size=n)-0.5)
        u = u*u
        s = pd.Series(u, index=index)
        clim, yws = dutils.runclim(s)

        yws = 5
        climc, datat = dutils.runclimcum(s, clim, yws)

        #import matplotlib.pyplot as plt
        #climc[['10%', '50%', '90%']].plot(); plt.show()
        #TODO

    def test_aggmonths(self):

        # Generate daily data with gaps
        index = pd.date_range('1950-01-01', '1950-12-31', freq='D')
        nval = len(index)
        val = np.random.uniform(size=nval)
        u = pd.Series(val, index=index)

        # -> 3 consecutive gaps
        idx = index >= datetime.datetime(1950, 2, 5)
        idx = idx & (index <= datetime.datetime(1950, 2, 7))
        u.loc[idx] = np.nan

        # -> 4 consecutive gaps
        idx = index >= datetime.datetime(1950, 3, 5)
        idx = idx & (index <= datetime.datetime(1950, 3, 8))
        u.loc[idx] = np.nan

        for d in [2, 5, 6, 8, 11, 20]:
            # -> 6 gaps
            dd = datetime.datetime(1950, 4, d)
            u.loc[dd] = np.nan

            # -> 6 gaps
            dd = datetime.datetime(1950, 5, d)
            u.loc[dd] = np.nan

        # -> one more gap
        dd = datetime.datetime(1950, 5, 22)
        u.loc[dd] = np.nan

        # Compute monthly and seasonal data
        out1 = dutils.aggmonths(u, nmonths=1)
        out2 = dutils.aggmonths(u, nmonths=3)

        # Test
        def _sum(x):
             return np.sum(x.values)

        expected1 = u.resample('MS', how=_sum)
        expected2 = out1 + out1.shift(-1) + out1.shift(-2)

        idxe = pd.notnull(expected1)
        self.assertTrue(np.allclose(expected1[idxe], out1[idxe]))

        idxe = pd.isnull(out1)
        idxo = (out1.index.month == 3) | (out1.index.month == 5)
        self.assertTrue(np.allclose(idxe, idxo))

        idxo = pd.notnull(out2)
        self.assertTrue(np.allclose(out2[idxo], expected2[idxo]))

    def test_atmospress(self):

        alt = 0
        p = dutils.atmospress(alt)
        self.assertTrue(np.allclose(p, 101325.))


        alt = 100
        p = dutils.atmospress(alt)
        self.assertTrue(np.allclose(p, 100130.800974))

        alt = 200
        p = dutils.atmospress(alt)
        self.assertTrue(np.allclose(p, 98950.6765392))


if __name__ == "__main__":
    unittest.main()
