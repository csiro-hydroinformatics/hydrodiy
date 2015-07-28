import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydata import baseflow
from hydata import hykiwis

class BaseflowTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BaseflowTestCase (hydata)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_baseflow(self):

        hyk = hykiwis.HyKiwis(False)

        attrs = hyk.get_tsattrs('410001')
        q = hyk.get_data(attrs, '1980-01-01', '1985-12-31')

        params = [0.95, 5, 0.95]

        bf1, BFI1 = baseflow.baseflow(q, params, method=1)
        bf2, BFI2 = baseflow.baseflow(q, params, method=2)
        bf3, BFI3 = baseflow.baseflow(q, params, method=3)

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(q.values, label='obs')
        ax.plot(bf1, label='Meth 1 (%0.2f)' % BFI1)
        ax.plot(bf1, label='Meth 2 (%0.2f)' % BFI2)
        ax.plot(bf1, label='Meth 3 (%0.2f)' % BFI3)

        ax.legend(loc='best')

        fp = '%s/bf.png' % self.FOUT
        fig.savefig(fp)

        #self.assertEqual(list(ts_data.astype(int)), 
        #        [13964, 14018, 15702, 16402, 15410])


if __name__ == "__main__":
    unittest.main()
