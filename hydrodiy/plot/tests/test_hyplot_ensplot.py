import os
import unittest

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from hydrodiy.plot.ensplot import MonthlyEnsplot
from hydrodiy.plot.ensplot import pitmetrics, pitplot

# Reset matplotlib to default
mpl.rcdefaults()

class MonthlyEnsplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MonthlyEnsplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

        dt = pd.date_range('1980-01-01', '2012-12-01', freq='MS')
        self.fcdates = dt
        nval = len(dt)
        self.obs = np.random.uniform(0, 1, nval)

        nens = 200
        self.fcst = np.random.uniform(0, 1, (nval, nens))


    def test_pitmetrics(self):
        ''' Test pitmetrics '''
        alpha, cr, pits = pitmetrics(self.obs, self.fcst)
        self.assertTrue(alpha>1)
        self.assertTrue(len(pits) == len(self.obs))


    def test_pitplot(self):
        ''' Test pitplot '''
        alpha, cr, pits = pitmetrics(self.obs, self.fcst)

        plt.close('all')
        fig, ax = plt.subplots()
        pitplot(pits, alpha, cr, ax)

        fp = os.path.join(self.fimg, 'pitplot.png')
        fig.savefig(fp)


    def test_monthplot(self):
        ''' Test all year plot '''

        plt.close('all')
        fig, ax = plt.subplots()

        mep = MonthlyEnsplot(self.obs, self.fcst, self.fcdates, fig)
        mep.monthplot(month=1, ax=ax, pit=True)

        fp = os.path.join(self.fimg, 'monthplot.png')
        fig.savefig(fp)


    def test_yearplot(self):
        ''' Test all year plot '''

        plt.close('all')
        fig = plt.figure()

        mep = MonthlyEnsplot(self.obs, self.fcst, self.fcdates, fig)
        mep.yearplot()

        fp = os.path.join(self.fimg, 'yearplot.png')
        mep.savefig(fp)





if __name__ == "__main__":
    unittest.main()
