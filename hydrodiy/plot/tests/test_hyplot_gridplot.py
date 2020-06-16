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
from hydrodiy.gis.oz import Oz, HAS_BASEMAP
from hydrodiy.plot.gridplot import GridplotConfig, gplot, gsmooth
from hydrodiy.plot.gridplot import VARNAMES, gbar

from hydrodiy.data.hywap import get_data
from hydrodiy.plot import putils
from hydrodiy.gis.grid import get_mask, Grid


class GridplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GridplotTestCase (hyplot)')

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


    def test_gconfig2str(self):
        for varname in VARNAMES:
            cfg = GridplotConfig(varname)
            str(cfg)


    def test_gsmooth(self):
        ''' Test grid smoother '''
        plt.close('all')
        fig, axs = plt.subplots(ncols=3, nrows=2)

        grd = self.grd.clone()
        grd.data[np.isnan(grd.data)] = -1
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
        ax.set_title('No mask applied with default smoothing')

        ax = axs[0, 2]
        sm = gsmooth(grd, sigma=10.)
        sm.plot(ax, cmap='Blues')
        ax.set_title('No mask applied with large smoothing')

        ax = axs[1, 2]
        sm = gsmooth(grd, self.mask, coastwin=100, sigma=10.)
        sm.plot(ax, cmap='Blues')
        ax.set_title('Mask applied with large smoothing')

        fig.set_size_inches((18, 12))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'gsmooth.png')
        fig.savefig(fp)


    def test_gplot(self):
        ''' Test gplot generation for different variables '''
        for varname in VARNAMES:
            print('Grid plot for {}'.format(varname))
            plt.close('all')
            fig = plt.figure()
            gs = GridSpec(nrows=2, ncols=2, \
                height_ratios=[2, 1], \
                width_ratios=[7, 1])

            ax = plt.subplot(gs[:,0])
            if HAS_BASEMAP:
                omap = Oz(ax=ax)
            else:
                omap = ax

            # Get config
            cfg = GridplotConfig(varname)

            # Set linewidth to see contours
            cfg.contour_linewidth = 0.8

            # generate data
            grd = self.grd.clone()
            y0, y1 = cfg.clevs[0], cfg.clevs[-1]
            grd.data = y0 + (y1-y0)*grd.data

            # Plot
            cont_gr, cont_lines, _, _ = gplot(grd, cfg, omap)
            cbar_ax = plt.subplot(gs[0, 1])
            dticks = True
            if re.search('decile|relative', varname):
                dticks = False

            gbar(cbar_ax, cfg, cont_gr, draw_ticks=dticks)

            fig.set_size_inches((7, 6))
            fig.tight_layout()
            fp = os.path.join(self.fimg, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


    def test_gplot_tweak_config(self):
        ''' Test tweaking plot config '''
        plt.close('all')
        fig = plt.figure()
        gs = GridSpec(nrows=2, ncols=2, \
            height_ratios=[2, 1], \
            width_ratios=[3, 1])

        ax = plt.subplot(gs[:,0])
        if HAS_BASEMAP:
            omap = Oz(ax=ax)
        else:
            omap = ax

        # Get config
        cfg = GridplotConfig('rainfall')
        cfg.cmap = 'Reds'
        cfg.clevs = [0, 50, 100, 150, 200]
        cfg.clevs_contour = [50, 150]
        cfg.contour_linewidth = 2

        # generate data
        grd = self.grd.clone()
        y0, y1 = cfg.clevs[0], cfg.clevs[-1]
        grd.data = y0 + (y1-y0)*grd.data

        # Plot
        cont_gr, cont_lines, _, _ = gplot(grd, cfg, omap)
        cbar_ax = plt.subplot(gs[0, 1])
        gbar(cbar_ax, cfg, cont_gr)

        fig.set_size_inches((7, 6))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'gridplot_tweak_config.png')
        fig.savefig(fp)


    def test_gbar_rect(self):
        ''' Test gplot generation with customised options '''

        cfg = GridplotConfig('rainfall')
        grd = self.grd
        y0, y1 = cfg.clevs[0], cfg.clevs[-1]
        grd.data = y0 + (y1-y0)*grd.data

        plt.close('all')

        fig = plt.figure()
        gs = GridSpec(nrows=2, ncols=6, \
            height_ratios=[2, 1], \
            width_ratios=[8, 1, 8, 1, 8, 1])

        for iplot in range(3):
            ax = plt.subplot(gs[:, 2*iplot])
            if HAS_BASEMAP:
                omap = Oz(ax=ax)
            else:
                omap = ax

            cont_gr, cont_lines, _, _ = gplot(grd, cfg, omap)

            cbar_ax = plt.subplot(gs[0, 2*iplot+1])
            if iplot == 0:
                gbar(cbar_ax, cfg, cont_gr, rect=[0.2, 0.1, 1, 1])
            elif iplot == 1:
                gbar(cbar_ax, cfg, cont_gr, rect=[0., 0., 0.6, 0.5])
            elif iplot == 2:
                gbar(cbar_ax, cfg, cont_gr, rect=[0.2, 0.1, 0.6, 0.5])

        fig.set_size_inches((20, 8))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'gbar_rect.png')
        fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
