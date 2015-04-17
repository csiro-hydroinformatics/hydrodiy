import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydata import hywap

class HyWapTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyWapTestCase (hydata)')

        FAWAP = '%s/awap' % os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(FAWAP): 
            os.mkdir(FAWAP)

        self.FAWAP = FAWAP

    def test_getgriddata(self):
       
        hya = hywap.HyWap()

        varname = 'rainfall'
        vartype = 'totals'
        dt = '1900-01-05'

        data, comment = hya.getgriddata(varname, vartype, dt)

    def test_writegriddata(self):
       
        hya = hywap.HyWap()

        F = self.FAWAP
        hya.set_awapdir(F)

        varname = 'rainfall'
        vartype = 'totals'
        dt = '1900-01-05'

        hya.writegriddata(varname, vartype, dt)


        import pdb; pdb.set_trace()

        #self.assertEqual(list(ts_data.astype(int)), 
        #        [13964, 14018, 15702, 16402, 15410])

if __name__ == "__main__":
    unittest.main()
