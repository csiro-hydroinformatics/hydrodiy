import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydrodiy.data import baseflow
from hydrodiy.data import hykiwis

class BaseflowTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BaseflowTestCase (hydata)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_baseflow(self):

        hyk = hykiwis.HyKiwis(False)

        attrs = hyk.get_tsattrs('410001')
        q = hyk.get_data(attrs, '1980-01-01', '1985-12-31')
        q = q.values.astype(np.float64)

        params = [0.01, 5, 0.95]

        bf1 = baseflow.baseflow(q, params, method=1)
        idx = pd.notnull(q) & (q>=0)
        BFI1 = np.sum(bf1[idx])/np.sum(q[idx])

        bf2 = baseflow.baseflow(q, params, method=2)
        BFI2 = np.sum(bf2[idx])/np.sum(q[idx])

        bf3 = baseflow.baseflow(q, params, method=3)
        BFI3 = np.sum(bf3[idx])/np.sum(q[idx])

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(q, label='obs')
        ax.plot(bf1, label='Meth 1 BFI = %0.2f' % BFI1)
        ax.plot(bf2, label='Meth 2 BFI = %0.2f' % BFI2)
        ax.plot(bf3, label='Meth 3 BFI = %0.2f' % BFI3)

        ax.legend(loc='best')

        fp = '%s/bf.png' % self.FOUT
        fig.savefig(fp)

        #self.assertEqual(list(ts_data.astype(int)),
        #        [13964, 14018, 15702, 16402, 15410])


if __name__ == "__main__":
    unittest.main()
