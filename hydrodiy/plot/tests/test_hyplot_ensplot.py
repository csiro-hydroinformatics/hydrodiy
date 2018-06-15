import os
import unittest

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from hydrodiy.plot.ensplot import MonthlyEnsplot
from hydrodiy.plot.ensplot import pitmetrics, pitplot, tsplot

# Reset matplotlib to default
mpl.rcdefaults()

class EnsplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> EnsplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

        dt = pd.date_range('1980-01-01', '2012-12-01', freq='MS')
        self.fcdates = dt
        nval = len(dt)
        nens = 200

        # Non censored forecasts
        self.obs = np.random.uniform(0, 1, nval)
        self.fcst = np.random.uniform(0, 1, (nval, nens))

        # Forecasts censored
        self.obs_sudo = np.clip(np.random.uniform(-0.25, 1, nval), 0, 1)
        self.fcst_sudo = np.clip(np.random.uniform(-0.25, 1, \
                                                (nval, nens)), 0, 1)


    def test_pitmetrics(self):
        ''' Test pitmetrics '''
        alpha, cr, pits, is_sudo = pitmetrics(self.obs, self.fcst)
        self.assertTrue(alpha>1)
        self.assertTrue(len(pits) == len(self.obs))


    def test_pitmetrics_sudo(self):
        ''' Test pitmetrics with sudo pits '''
        alpha, cr, pits, is_sudo = pitmetrics(self.obs_sudo, self.fcst_sudo)
        self.assertTrue(len(pits) == len(self.obs_sudo))
        self.assertTrue(np.sum(is_sudo) > 0)


    def test_tsplot(self):
        ''' Test tsplot '''
        plt.close('all')
        fig, ax = plt.subplots()
        x = tsplot(self.obs, self.fcst, ax, \
                    show_pit=True, show_scatter=True, \
                    line='mean')

        fig.set_size_inches((30, 7))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'tsplot.png')
        fig.savefig(fp)


    def test_pitplot(self):
        ''' Test pitplot '''
        alpha, cr, pits, sudo = pitmetrics(self.obs, self.fcst)

        plt.close('all')
        fig, ax = plt.subplots()
        pitplot(pits, sudo, alpha, cr, ax)

        fp = os.path.join(self.fimg, 'pitplot.png')
        fig.savefig(fp)


    def test_pitplot_sudo(self):
        ''' Test pitplot with sudo pits'''
        alpha, cr, pits, sudo = pitmetrics(self.obs_sudo, self.fcst_sudo)

        plt.close('all')
        fig, ax = plt.subplots()
        pitplot(pits, sudo, alpha, cr, ax)

        fp = os.path.join(self.fimg, 'pitplot_sudo.png')
        fig.savefig(fp)


    def test_monthplot(self):
        ''' Test one month plot '''

        plt.close('all')
        fig, ax = plt.subplots()

        mep = MonthlyEnsplot(self.obs, self.fcst, self.fcdates, fig)
        mep.monthplot(month=1, ax=ax)

        fp = os.path.join(self.fimg, 'monthplot.png')
        fig.savefig(fp)


    def test_monthplot_sudo(self):
        ''' Test one month plot with sudo pits '''

        plt.close('all')
        fig, ax = plt.subplots()

        mep = MonthlyEnsplot(self.obs_sudo, self.fcst_sudo, self.fcdates, fig)
        mep.monthplot(month=1, ax=ax)

        fp = os.path.join(self.fimg, 'monthplot_sudo.png')
        fig.savefig(fp)



    def test_yearplot(self):
        ''' Test all year plot '''

        plt.close('all')
        fig = plt.figure()

        mep = MonthlyEnsplot(self.obs, self.fcst, self.fcdates, fig)
        perf = mep.yearplot()

        perf = pd.DataFrame(perf).T
        self.assertEqual(perf.shape, (13, 3))
        self.assertEqual(list(perf.columns), ['R2', 'alpha', 'crps_ss'])

        fp = os.path.join(self.fimg, 'yearplot.png')
        mep.savefig(fp)



    def test_yearplot_sudo(self):
        ''' Test all year plot with sudo '''

        plt.close('all')
        fig = plt.figure()

        mep = MonthlyEnsplot(self.obs_sudo, self.fcst_sudo, self.fcdates, fig)
        mep.yearplot()

        fp = os.path.join(self.fimg, 'yearplot_sudo.png')
        mep.savefig(fp)





if __name__ == "__main__":
    unittest.main()
