import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydrodiy.plot.simplot import Simplot
from hydrodiy.gis import oz

class SimplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SimplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_sim_daily(self):

        dt = pd.date_range('2010-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        plt.close('all')
        sm = Simplot(obs, sim, sim_name='bidule')

        sm.add_sim(sim2, sim_name='truc')
        sm.draw()

        fp = os.path.join(self.ftest, 'simplot_daily.png')
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


if __name__ == "__main__":
    unittest.main()
