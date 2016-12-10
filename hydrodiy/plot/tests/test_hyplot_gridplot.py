import os
import math
import unittest

from  datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.data.hywap import get_data
from hydrodiy.plot import putils
from hydrodiy.gis.grid import get_ref_grid
from hydrodiy.gis.oz import Oz

from hydrodiy.plot.gridplot import get_lim, get_config, plot, smooth
from hydrodiy.plot.gridplot import VARNAMES, plot_colorbar

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

    def test_get_lim(self):
        regions = ['CAPEYORK', 'AUS', 'COASTALNSW', \
                    'MDB', 'VIC+TAS', 'PERTH', 'QLD']

        for region in regions:
            xlim, ylim = get_lim(region)

        try:
            get_lim('XXX')
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Region XXX'))


    def test_get_config(self):
        for varname in VARNAMES:
            cfg = get_config(varname)

            k1 = np.sort(cfg.keys())
            k2 = np.array(['clevs', 'clevs_tick_labels', 'clevs_ticks', \
                    'cmap', 'linecolor', \
                    'linewidth', 'norm'])
            ck = np.all([k1[i] == k2[i] for i in range(len(k1))])
            self.assertTrue(ck)


    def test_smooth(self):
        plt.close('all')
        fig, axs = plt.subplots(ncols=3)
        self.grd.plot(axs[0], cmap='Blues')

        sm = smooth(self.grd, self.mask)
        sm.plot(axs[1], cmap='Blues')

        sm = smooth(self.grd, self.mask, sigma=1.)
        sm.plot(axs[2], cmap='Blues')

        fig.set_size_inches((18, 6))
        fig.tight_layout()
        fp = os.path.join(self.ftest, 'smooth.png')
        fig.savefig(fp)


    def test_plot(self):
        plt.close('all')
        putils.set_mpl('white')

        sm = smooth(self.grd, self.mask)

        for varname in VARNAMES:
            cfg = get_config(varname)
            fig, ax = plt.subplots()

            om = Oz(ax=ax)
            bm = om.get_map()

            plot(sm, bm, cfg)

            fig.tight_layout()
            fp = os.path.join(self.ftest, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
