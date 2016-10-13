import os
import unittest
import itertools

import numpy as np
import datetime
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid import axes_size

from hydrodiy.data import hywap

if hywap.HAS_BASEMAP:
    from hydrodiy.gis import oz

class HyWapTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyWapTestCase (hydata)')

        FAWAP = '%s/awap' % os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(FAWAP):
            os.mkdir(FAWAP)

        self.FAWAP = FAWAP

    def test_get_data(self):

        hya = hywap.HyWap()

        dt = '2015-02-01'

        vn = hywap.VARIABLES
        ts = hywap.TIMESTEPS

        for varname, timestep in itertools.product(vn.keys(), ts):

            for v in vn[varname]:
                vartype = v['type']

                data, comment, header = hya.get_data(varname, vartype,
                                            timestep, dt)

                nr = int(header['nrows'])
                nc = int(header['ncols'])

                self.assertEqual(data.shape, (nr, nc))


    def test_get_cellcoords(self):

        hya = hywap.HyWap()

        varname = 'rainfall'
        vartype = 'totals'
        ts = 'month'
        dt = '2015-03-01'

        data, comment, header = hya.get_data(varname, vartype, ts, dt)

        cellids, llongs, llats = hywap.get_cellcoords(header)

        nr = header['nrows']
        nc = header['ncols']

        self.assertEqual(cellids.shape, (nr, nc))
        self.assertEqual(llongs.shape, (nr, nc))
        self.assertEqual(llats.shape, (nr, nc))

        xll = header['xllcorner']
        yll = header['yllcorner']
        self.assertEqual(cellids[-1,0], '%0.2f_%0.2f' % (xll, yll))


    def test_plot(self):

        hya = hywap.HyWap()

        dt = '2015-02-01'

        vn = hywap.VARIABLES
        ts = hywap.TIMESTEPS

        if hywap.HAS_BASEMAP:

            for varname, timestep in itertools.product(vn.keys(), ts):

                cfg = hywap.get_plotconfig(None, varname)

                if varname == 'rainfall':
                    cfg['norm'] = mpl.colors.SymLogNorm(
                            linthresh = 1,
                            vmin = 0.,
                            vmax = cfg['clevs'][-1])

                for v in vn[varname]:
                    vartype = v['type']

                    data, comment, header = hya.get_data(varname, vartype,
                                                timestep, dt)

                    fig, ax = plt.subplots()

                    om = oz.Oz(ax = ax)
                    om.drawcoast(linestyle='-')
                    om.drawstates(linestyle='--')

                    cs = hya.plot(data, header, om, config=cfg)

                    sz = axes_size.AxesY(ax, 0.6)
                    sz = '10%'
                    hywap.plot_cbar(fig, ax, cs,
                        position='right', size=sz, pad=0.1)

                    ax.set_title('%s - %s' % (varname, dt))

                    fp = '%s/%s_%s_%s.png' % (self.FAWAP, varname, timestep, vartype)
                    fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
