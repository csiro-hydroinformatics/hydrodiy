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
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

        varname = 'rainfall'
        vartype = 'totals'
        timestep = 'month'
        dt = datetime(2015, 1, 1)

        self.mask = get_mask('AWAP')

        # Generate random rainfall data
        grd = self.mask.clone(np.float64)
        xx, yy = np.meshgrid(np.linspace(0, 2*math.pi, grd.ncols), \
                                 np.linspace(0, 2*math.pi, grd.nrows))
        grd.data = (np.cos(3*xx+6*yy)+1)/2
        grd.data[self.mask.data == 0] = np.nan
        self.grd = grd

    def test_get_gconfig(self):
        for varname in VARNAMES:
            cfg = GridplotConfig(varname)
            cfg.is_valid()


    def test_gsmooth(self):
        ''' Test grid smoother '''
        plt.close('all')
        fig, axs = plt.subplots(ncols=3, nrows=2)

        grd = self.grd.clone()
        grd.data += np.random.uniform(-0.8, 0.8, size=grd.data.shape)

        ax = axs[0, 0]
        grd.plot(ax, cmap='Blues')
        ax.set_title('Raw gridded data')

        ax = axs[1, 0]
        sm = gsmooth(grd, self.mask, sigma=1e-5)
        sm.plot(ax, cmap='Blues')
        ax.set_title('Mask applied with insignificant smoothing')

        ax = axs[0, 1]
        sm = gsmooth(grd)
        sm.plot(ax, cmap='Blues')
        ax.set_title('Default smoothing options')

        ax = axs[1, 1]
        sm = gsmooth(grd, self.mask)
        sm.plot(ax, cmap='Blues')
        ax.set_title('Mask applied with default smoothing')

        ax = axs[0, 2]
        sm = gsmooth(grd, sigma=10.)
        sm.plot(ax, cmap='Blues')
        ax.set_title('No mask applied with large smoothing')

        ax = axs[1, 2]
        sm = gsmooth(grd, self.mask, sigma=10.)
        sm.plot(ax, cmap='Blues')
        ax.set_title('Mask applied with large smoothing')

        fig.set_size_inches((18, 12))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'gsmooth.png')
        fig.savefig(fp)


    def test_gplot(self):
        ''' Test gplot generation for different variables '''
        plt.close('all')

        for varname in VARNAMES:
            print('Grid plot for {}'.format(varname))
            fig = plt.figure()
            gs = GridSpec(nrows=2, ncols=2, \
                height_ratios=[2, 1], \
                width_ratios=[3, 1])

            ax = plt.subplot(gs[:,0])
            om = Oz(ax=ax)
            bm = om.map

            # Get config
            cfg = GridplotConfig(varname)

            # generate data
            grd = self.grd.clone()
            y0, y1 = cfg.clevs[0], cfg.clevs[-1]
            grd.data = y0 + (y1-y0)*grd.data

            # Plot
            cont_gr, cont_lines = gplot(grd, om.map, cfg)
            cbar_ax = plt.subplot(gs[0, 1])
            gbar(cbar_ax, cfg, cont_gr)

            fig.set_size_inches((7, 6))
            fig.tight_layout()
            fp = os.path.join(self.fimg, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


    def test_gbar_options(self):
        ''' Test gplot generation with customised options '''

        grd = self.grd
        cfg = GridplotConfig('soil-moisture')

        plt.close('all')

        fig = plt.figure()
        gs = GridSpec(nrows=6, ncols=2, \
            height_ratios=[2, 1, 2, 1, 2, 1], \
            width_ratios=[5, 1])

        # Aspect
        for iopt in range(3):
            ax = plt.subplot(gs[2*iopt:2*iopt+2, 0])
            om = Oz(ax=ax)
            bm = om.map
            cont_gr, cont_lines = gplot(grd, om.map, cfg)

            cbar_ax = plt.subplot(gs[2*iopt, 1])
            if iopt == 0:
                gbar(cbar_ax, cfg, cont_gr, aspect=20)
            elif iopt == 1:
                gbar(cbar_ax, cfg, cont_gr, aspect=0.5)
            elif iopt == 2:
                gbar(cbar_ax, cfg, cont_gr, fraction=1.5, \
                                    location='right', pad=0.5)

        fig.set_size_inches((8, 18))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'gbar_options.png')
        fig.savefig(fp)


if __name__ == "__main__":

    if import_ok:
        unittest.main()
