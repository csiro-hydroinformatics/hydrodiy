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

    def test_oz1(self):
        om = Oz()
        plt.close()
        fig, ax = plt.subplots()
        npt = 10
        x = np.random.normal(loc=133, scale=20, size=npt)
        y = np.random.normal(loc=-25, scale=20, size=npt)
        ax.plot(x,y,'bo')
        om.plot_states(ax)
        om.set_rangeoz(ax)
        putils.footer(fig) 
        fp = '%s/oz_plot1.png'%self.FOUT
        plt.savefig(fp)

    def test_oz2(self):
        om = Oz()
        plt.close()
        fig, ax = plt.subplots()
        npt = 10
        om.plot_states(ax)
        data = pd.Series(['%0.3f'%a for a in np.random.random(3)], 
                index=['ACT', 'NT', 'SA'])
        om.plot_states_data(ax, data)        
        om.set_rangeoz(ax)
        putils.footer(fig) 
        fp = '%s/oz_plot2.png'%self.FOUT
        plt.savefig(fp)

    def test_oz3(self):
        om = Oz()
        plt.close()
        fig, ax = plt.subplots()
        om.plot_ozpng(ax)
        om.plot_states(ax)
        om.set_rangeoz(ax)
        putils.footer(fig) 
        fp = '%s/oz_plot3.png'%self.FOUT
        plt.savefig(fp)


if __name__ == "__main__":
    unittest.main()
