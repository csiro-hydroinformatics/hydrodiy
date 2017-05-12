import os
import unittest

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from hydrodiy.plot.simplot import Simplot
from hydrodiy.gis import oz

# Reset matplotlib to default
mpl.rcdefaults()

class SimplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SimplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_sim_daily(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = Simplot(obs, sim, sim_name='bidule')

        sm.add_sim(sim2, name='truc')
        sm.draw()

        fp = os.path.join(self.ftest, 'simplot_daily.png')
        sm.savefig(fp)


    def test_nfloods(self):
        dt = pd.date_range('2000-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = Simplot(obs, sim, sim_name='bidule', nfloods=10)
        sm.draw()

        fp = os.path.join(self.ftest, 'simplot_nfloods.png')
        sm.savefig(fp)



    def test_sim_monthly(self):
        dt = pd.date_range('2000-01-01', '2015-12-01', freq='MS')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = Simplot(obs, sim)
        sm.draw()

        fp = os.path.join(self.ftest, 'simplot_monthly.png')
        sm.savefig(fp)


    def test_axis(self):
        dt = pd.date_range('1970-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = Simplot(obs, sim, sim_name='bidule')

        axb, axa, axfd, axfdl, axs, axf = sm.draw()

        axb.set_title('T1')
        axa.set_title('T2')
        axfd.set_title('T3')
        axfdl.set_title('T4')
        axs.set_title('T5')
        for ax in axf:
            ax.set_ylim([0, 1])

        fp = os.path.join(self.ftest, 'simplot_axis.png')
        sm.savefig(fp)


if __name__ == "__main__":
    unittest.main()
