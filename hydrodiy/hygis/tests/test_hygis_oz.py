import os
import unittest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from hygis.oz import Oz
from hyplot import putils

class OzTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> OzTestCase')
        FTEST, ozfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_oz0(self):

        plt.close('all')

        om = Oz()
        om.drawcoast()
        om.drawstates()
      
        fp = '%s/oz_plot0.png'%self.FOUT
        plt.savefig(fp)

    def test_oz1(self):

        plt.close('all')
        fig, ax = plt.subplots()

        om = Oz(ax)
        om.drawcoast()
        om.drawstates()

        npt = 100
        x = np.random.normal(loc=133, scale=20, size=npt)
        y = np.random.normal(loc=-25, scale=20, size=npt)
        om.plot(x, y, 'ro')
       
        putils.footer(fig) 
        fp = '%s/oz_plot1.png'%self.FOUT
        fig.savefig(fp)

    def test_oz2(self):

        plt.close('all')

        om = Oz()
        om.drawrelief()
        om.drawcoast()
        om.drawstates()
      
        fp = '%s/oz_plot2.png'%self.FOUT
        plt.savefig(fp)

if __name__ == "__main__":
    unittest.main()
