import os, re, math
import unittest
from  datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.data.hywap import get_data
from hydrodiy.plot import putils
from hydrodiy.gis.grid import get_ref_grid
from hydrodiy.gis.oz import Oz

from hydrodiy.plot.gridplot import get_gconfig, gplot, gsmooth
from hydrodiy.plot.gridplot import VARNAMES, gplot_colorbar

class GridplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GridplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        varname = 'rainfall'
        vartype = 'totals'
        timestep = 'month'
        dt = datetime(2015, 1, 1)
        self.grd = get_data(varname, vartype,
                                            timestep, dt)

        self.mask = get_ref_grid('AWAP')

    def test_get_gconfig(self):
        for varname in VARNAMES:
            cfg = get_gconfig(varname)

            k1 = np.sort(cfg.keys())
            k2 = np.array(['clevs', 'clevs_tick_labels', 'clevs_ticks', \
                    'cmap', 'linecolor', \
                    'linewidth', 'norm'])
            ck = np.all([k1[i] == k2[i] for i in range(len(k1))])
            self.assertTrue(ck)


    def test_gsmooth(self):
        plt.close('all')
        fig, axs = plt.subplots(ncols=3, nrows=2)

        self.grd.plot(axs[0, 0], cmap='Blues')

        sm = gsmooth(self.grd, self.mask, sigma=1e-5)
        sm.plot(axs[1, 0], cmap='Blues')

        sm = gsmooth(self.grd)
        sm.plot(axs[0, 1], cmap='Blues')

        sm = gsmooth(self.grd, self.mask)
        sm.plot(axs[1, 1], cmap='Blues')

        sm = gsmooth(self.grd, sigma=50.)
        sm.plot(axs[0, 2], cmap='Blues')

        sm = gsmooth(self.grd, self.mask, sigma=50.)
        sm.plot(axs[1, 2], cmap='Blues')

        fig.set_size_inches((18, 12))
        fig.tight_layout()
        fp = os.path.join(self.ftest, 'gsmooth.png')
        fig.savefig(fp)


    def test_gplot(self):
        plt.close('all')
        putils.set_mpl('white')

        sm = gsmooth(self.grd, self.mask)

        for varname in VARNAMES:
            cfg = get_gconfig(varname)
            fig, ax = plt.subplots()

            om = Oz(ax=ax)
            bm = om.get_map()

            sm2 = sm
            if re.search('decile|moisture', varname):
                sm2 = sm.clone()
                dt = sm2.data
                sm2.data = dt/np.nanmax(dt)

            gplot(sm2, bm, cfg)

            fig.tight_layout()
            fp = os.path.join(self.ftest, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
