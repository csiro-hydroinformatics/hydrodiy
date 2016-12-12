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

from hydrodiy.plot.gridplot import GridplotConfig, gplot, gsmooth
from hydrodiy.plot.gridplot import VARNAMES, gbar

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
        return
        plt.close('all')
        putils.set_mpl('white')

        sm = gsmooth(self.grd, self.mask)

        for varname in VARNAMES:
            cfg = GridplotConfig(varname)
            fig, ax = plt.subplots()

            om = Oz(ax=ax)
            bm = om.get_map()

            sm2 = sm.clone()
            if re.search('decile|moisture', varname):
                dt = sm2.data
                sm2.data = dt/np.nanmax(dt)
            elif re.search('effective', varname):
                sm2.data = sm2.data - 50

            gplot(sm2, bm, cfg)

            fig.tight_layout()
            fp = os.path.join(self.ftest, 'gridplot_{0}.png'.format(varname))
            fig.savefig(fp)


    def test_gbar(self):
        plt.close('all')
        putils.set_mpl('black')

        grd = self.grd
        grd.data = grd.data - 20
        sm = gsmooth(grd, self.mask)

        varname = 'effective-rainfall'
        fig, ax = plt.subplots()

        om = Oz(ax=ax)

        cfg = GridplotConfig(varname)
        import pdb; pdb.set_trace()
        contf = gplot(sm, om.get_map(), cfg)
        gbar(fig, ax, cfg, contf)

        fig.tight_layout()
        fp = os.path.join(self.ftest, 'gridplot_colorbar_{0}.png'.format(varname))
        fig.savefig(fp)



if __name__ == "__main__":
    unittest.main()
