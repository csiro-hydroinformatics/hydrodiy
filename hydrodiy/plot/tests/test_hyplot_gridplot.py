import os, re, math
import unittest
from  datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Skip if package cannot be imported (circleci build)
import_error = True
try:
    from hydrodiy.gis.oz import Oz
    from hydrodiy.plot.gridplot import GridplotConfig, gplot, gsmooth
    from hydrodiy.plot.gridplot import VARNAMES, gbar

    import_error = False
except ImportError:
    pass

from hydrodiy.data.hywap import get_data
from hydrodiy.plot import putils
from hydrodiy.gis.grid import get_mask, Grid


class GridplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GridplotTestCase (hyplot)')
        if import_error:
            self.skipTest('Import error')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        varname = 'rainfall'
        vartype = 'totals'
        timestep = 'month'
        dt = datetime(2015, 1, 1)

        self.mask = get_mask('AWAP')

        # Generate random rainfall data
        grd = self.mask.clone()
        nval = grd.nrows*grd.ncols
        data = np.maximum(np.random.normal(loc=100, scale=50, size=nval), 0)
        grd.data = data.reshape((grd.nrows, grd.ncols))

        self.grd = grd

    def test_get_gconfig(self):
        for varname in VARNAMES:
            cfg = GridplotConfig(varname)
            cfg.is_valid()


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
            fig = plt.figure()
            gs = GridSpec(nrows=3, ncols=3, \
                height_ratios=[1, 4, 1], \
                width_ratios=[6, 1, 1])

            ax = plt.subplot(gs[:,0])
            om = Oz(ax=ax)
            bm = om.map

            sm2 = sm.clone()
            if re.search('decile|moisture', varname):
                dt = sm2.data
                sm2.data = dt/np.nanmax(dt)
            elif re.search('effective', varname):
                sm2.data = sm2.data - 50

            cfg = GridplotConfig(varname)
            contf = gplot(sm, om.map, cfg)

            cbar_ax = plt.subplot(gs[1, 2])
            gbar(cbar_ax, cfg, contf)

            fig.tight_layout()
            fp = os.path.join(self.ftest, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


if __name__ == "__main__":

    if import_ok:
        unittest.main()
