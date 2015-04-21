import os
import unittest
import numpy as np
import datetime
import pandas as pd

import matplotlib.pyplot as plt

from hydata import hywap
from hygis import oz

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

        data, comment, header = hya.getgriddata(varname, vartype, dt)

        self.assertEqual(data.shape, (691, 886))

    def test_writegriddata(self):
       
        hya = hywap.HyWap()

        F = self.FAWAP
        hya.set_awapdir(F)

        varname = 'rainfall'
        vartype = 'totals'
        dt = '1900-01-05'

        fdata = hya.writegriddata(varname, vartype, dt)

        self.assertTrue(os.path.exists(fdata))

    def test_getcoord(self):
        
        hya = hywap.HyWap()

        varname = 'rainfall'
        vartype = 'totals'
        dt = '2015-04-17'

        data, comment, header = hya.getgriddata(varname, vartype, dt)

        cellnum, llongs, llats = hya.getcoords(header)

        self.assertEqual(cellnum.shape, (691, 886))
        self.assertEqual(llongs.shape, (691, 886))
        self.assertEqual(llats.shape, (691, 886))

    def test_plotdata(self):
        
        hya = hywap.HyWap()

        varname = 'rainfall'
        vartype = 'totals'
        dt = '2015-04-17'

        data, comment, header = hya.getgriddata(varname, vartype, dt)

        fig, ax = plt.subplots()

        hya.plotdata(data, header, ax)

        ax.set_title('%s - %s' % (varname, dt))

        fp = '%s/rainfallsurf.png' % self.FAWAP
        fig.savefig(fp)

if __name__ == "__main__":
    unittest.main()
