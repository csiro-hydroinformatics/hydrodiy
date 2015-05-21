import os
import unittest
import itertools

import numpy as np
import datetime
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid import axes_size

from hydata import hywap

if hywap.has_basemap:
    from hygis import oz

class HyWapTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyWapTestCase (hydata)')

        FAWAP = '%s/awap' % os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(FAWAP): 
            os.mkdir(FAWAP)

        self.FAWAP = FAWAP

    def test_getgriddata(self):
       
        hya = hywap.HyWap()

        dt = '2015-02-01'

        vn = hya.variables
        ts = hya.timesteps

        for varname, timestep in itertools.product(vn.keys(), ts):

            for v in vn[varname]:
                vartype = v['type']

                data, comment, header = hya.getgriddata(varname, vartype, 
                                            timestep, dt)

                nr = int(header['nrows'])
                nc = int(header['ncols'])

                self.assertEqual(data.shape, (nr, nc))

    def test_savegriddata(self):
       
        hya = hywap.HyWap()

        F = self.FAWAP
        hya.set_awapdir(F)

        varname = 'rainfall'
        vartype = 'totals'
        ts = 'month'
        dt = '1900-01-01'

        fdata = hya.savegriddata(varname, vartype, ts, dt)

        self.assertTrue(os.path.exists(fdata))

    def test_getcoord(self):
        
        hya = hywap.HyWap()

        varname = 'rainfall'
        vartype = 'totals'
        ts = 'month'
        dt = '2015-03-01'

        data, comment, header = hya.getgriddata(varname, vartype, ts, dt)

        cellids, llongs, llats = hya.getcoords(header)

        nr = int(header['nrows'])
        nc = int(header['ncols'])

        self.assertEqual(cellids.shape, (nr, nc))
        self.assertEqual(llongs.shape, (nr, nc))
        self.assertEqual(llats.shape, (nr, nc))

        xll = float(header['xllcenter'])
        yll = float(header['yllcenter'])
        self.assertEqual(cellids[-1,0], '%0.2f_%0.2f' % (xll, yll))

    def test_plot(self):
        
        hya = hywap.HyWap()

        dt = '2015-02-01'

        vn = hya.variables
        ts = hya.timesteps

        if hywap.has_basemap:

            for varname, timestep in itertools.product(vn.keys(), ts):
                
                plotconfig = hya.default_plotconfig(None, varname)
                    
                if varname == 'rainfall':
                    plotconfig['norm'] = mpl.colors.SymLogNorm(
                            linthresh = 1,
                            vmin = 0., 
                            vmax = plotconfig['clevs'][-1])

                for v in vn[varname]:
                    vartype = v['type']
 
                    data, comment, header = hya.getgriddata(varname, vartype, 
                                                timestep, dt)

                    fig, ax = plt.subplots()

                    cs = hya.plot(data, header, ax, plotconfig=plotconfig)

                    sz = axes_size.AxesY(ax, 0.6)
                    sz = '10%'
                    hya.plot_cbar(fig, ax, cs, 
                        position='right', size=sz, pad=0.1)

                    ax.set_title('%s - %s' % (varname, dt))

                    fp = '%s/%s_%s_%s.png' % (self.FAWAP, varname, timestep, vartype)
                    fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
