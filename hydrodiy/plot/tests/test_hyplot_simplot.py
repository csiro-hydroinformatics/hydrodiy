import os
import unittest

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.plot import simplot
from hydrodiy.plot import putils

# Reset matplotlib to default
mpl.rcdefaults()

class SimplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SimplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)


    def test_sim_daily(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule')

        sm.add_sim(sim2, name='truc')
        axb, axa, axfd, axfdl, axs, axf = sm.draw()

        fp = os.path.join(self.fimg, 'simplot_daily.png')
        sm.savefig(fp)


    def test_sim_daily_samefloodyscale(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule', samefloodyscale=True)

        sm.add_sim(sim2, name='truc')
        axb, axa, axfd, axfdl, axs, axf = sm.draw()
        ylims = []
        for ax in axf:
            ylims.append(ax.get_ylim())
        ylims = np.array(ylims)
        self.assertTrue(np.all(np.std(ylims, 0) < 1e-10))

        fp = os.path.join(self.fimg, 'simplot_daily_samefloodyscale.png')
        sm.savefig(fp)


    def test_nfloods(self):
        dt = pd.date_range('2000-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule', nfloods=10)
        sm.draw()

        fp = os.path.join(self.fimg, 'simplot_nfloods.png')
        sm.savefig(fp)


    def test_sim_monthly(self):
        dt = pd.date_range('2000-01-01', '2015-12-01', freq='MS')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim)
        sm.draw()

        fp = os.path.join(self.fimg, 'simplot_monthly.png')
        sm.savefig(fp)


    def test_axis(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule')

        axb, axa, axfd, axfdl, axs, axf = sm.draw()

        axb.set_title('T1')
        axa.set_title('T2')
        axfd.set_title('T3')
        axfdl.set_title('T4')
        axs.set_title('T5')
        for ax in axf:
            ax.set_ylim([0, 1])

        fp = os.path.join(self.fimg, 'simplot_axis.png')
        sm.savefig(fp)


    def test_options_fdc_zoom(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule', \
                    fdc_zoom_xlim=[0., 0.2], \
                    fdc_zoom_ylog=False)

        axb, axa, axfd, axfdl, axs, axf = sm.draw()

        fp = os.path.join(self.fimg, 'simplot_fdc_zoom.png')
        sm.savefig(fp)


    def test_color_scheme(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        sch = simplot.COLOR_SCHEME
        cols = putils.cmap2colors(3, 'Spectral')
        simplot.COLOR_SCHEME = cols

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = simplot.Simplot(obs, sim, sim_name='bidule')
        axb, axa, axfd, axfdl, axs, axf = sm.draw()

        fp = os.path.join(self.fimg, 'simplot_colors.png')
        sm.savefig(fp)

        simplot.COLOR_SCHEME = sch


if __name__ == "__main__":
    unittest.main()
