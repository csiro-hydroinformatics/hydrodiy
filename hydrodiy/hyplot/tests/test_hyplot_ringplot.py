import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from hyplot.ringplot import RingPlot
from hygis.oz import Oz

class RingPlotTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> RingPlotTestCase (hyplot)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_match(self):
        plt.close()
        fig = plt.figure()
        rg = RingPlot(fig, ncols=4, nrows=3, 
                centreaxe_aspect='equal', 
                centreaxe_overlap=0.0, 
                ringaxe_axis_off=False, 
                ringaxe_border=0.06)

        nval = 10
        x = np.random.uniform(0, 2, size=nval)
        y = np.random.uniform(0, 1, size=nval)
        rg.centreaxe.plot(x,y, '+')
        rg.centreaxe.set_xlim((0,2))
        rg.centreaxe.set_ylim((0,1))
        mapping = rg.match_pts(x, y)        

        for idx, xx, yy in zip(mapping, x, y):
            rg.centreaxe.text(xx, yy, idx, va='bottom')
            rg.draw_line(idx[0], idx[1], xx, yy)

        fp = '%s/match.png'%self.FOUT
        fig.savefig(fp)

    def test_surrounding_1(self):
        plt.close()
        fig = plt.figure()
        rg = RingPlot(fig, ncols=4, nrows=3, 
                            centreaxe_overlap=0.1)
        x = np.linspace(0,1,50)
        rg.centreaxe.plot(x, x*np.exp(-x), '+')
        for idx, row in rg.ringaxes.iterrows():
            e = np.random.exponential(size=10)
            row['ax'].bar(range(10), e, 0.8)
            
        fp = '%s/surrounding1.png'%self.FOUT
        fig.savefig(fp)

    def test_surrounding_2(self):
        plt.close()
        fig = plt.figure()
        rg = RingPlot(fig, ncols=4, nrows=3, 
                centreaxe_aspect='equal', 
                centreaxe_overlap=0.2, 
                ringaxe_border=0.06)
        om = Oz()
        om.plot_items(rg.centreaxe, ['TAS'])
        
        xlim = rg.centreaxe.get_xlim()
        ylim = rg.centreaxe.get_ylim()
    
        npts = 10
        x = np.random.uniform(xlim[0], xlim[1], size=npts) 
        y = np.random.uniform(ylim[0], ylim[1], size=npts) 
        mapping = rg.match_pts(x, y)        

        for idx, xx, yy in zip(mapping, x, y):
            e = np.random.exponential(size=10)
            ax = rg.ringaxes.loc[(idx[0], idx[1]),'ax']
            ax.bar(range(10), e, 0.8)
            rg.draw_line(idx[0], idx[1], xx, yy)

        fp = '%s/surrounding2.png'%self.FOUT
        fig.savefig(fp)

if __name__ == "__main__":
    unittest.main()
