import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydrodiy.data import dutils

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


    def test_atmpressure(self):

        alt = 0
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 101325.))


        alt = 100
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 100130.800974))

        alt = 200
        p = dutils.atmpressure(alt)
        self.assertTrue(np.allclose(p, 98950.6765392))


if __name__ == "__main__":
    unittest.main()
