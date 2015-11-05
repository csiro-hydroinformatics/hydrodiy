import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hyplot.simplot import Simplot
from hygis import oz

class SimplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SimplotTestCase (hyplot)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_sim(self):

        plt.close('all')
        fig = plt.figure()

        dt = pd.date_range('2010-01-01', '2015-12-01')
        nval = len(dt)

        obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
        sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

        sm = Simplot(fig, obs, sim)
        sm.draw()

        fp = '%s/simplot.png' % self.FOUT
        fig.set_size_inches((15, 15))
        sm.gs.tight_layout(fig)
        fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
