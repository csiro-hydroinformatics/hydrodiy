import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hyplot import ninoplot
from hyplot import putils

class NinoPlotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> NinoPlotTestCase (hyplot)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST


    def test_topplot(self):

        idx = pd.date_range('1970-01-01', '2015-05-01', freq='MS')

        data = pd.Series(np.random.uniform(-10, 10, size=len(idx)), 
                index=idx, name='data top')

        plt.close('all')

        npl = ninoplot.NinoPlot()

        ylim = [-11, 11]
        yticks = [-5, 0, 5]

        npl.toplot(data, ylim, yticks=yticks, ygrid=True)

        fp = '%s/ninoplot_top.png'%self.FOUT
        npl.savefig(fp)


    def test_bottom_line(self):

        idx = pd.date_range('1970-01-01', '2015-05-01', freq='MS')

        data = pd.DataFrame({'data': np.random.uniform(0, 100, size=len(idx))}, 
                index=idx)
        data['highlight'] = [1 == (i%10)/10 for i in range(len(data))]


        plt.close('all')

        npl = ninoplot.NinoPlot()

        ylim = [-10, 120]
        yticks = [0, 50, 100]

        means = npl.bottomplot_line(data, ylim,
            yticks = yticks, 
            ygrid=True)

        npl.bottomplot_average(ylim, colors='',
            title='average',
            label='av.')

        fp = '%s/ninoplot_bottom_line.png'%self.FOUT
        npl.savefig(fp)


    def test_bottom_bars(self):

        idx = pd.date_range('1970-01-01', '2015-05-01', freq='MS')

        data = pd.DataFrame({'data1': np.random.uniform(0, 100, size=len(idx))}, 
                index=idx)
        data['data2'] = 100 - data['data1']

        plt.close('all')

        npl = ninoplot.NinoPlot(color_spines='none')

        ylim = [0, 100]
        yticks = np.arange(0, 120, 20)

        colors = putils.wafari_tercile_colors[:2]

        means = npl.bottomplot_bars(data, colors, ylim,
            yticks = yticks, 
            ygrid=False)

        npl.bottomplot_average(ylim, colors,
            title='average',
            label='av.',
            means = means)

        fp = '%s/ninoplot_bottom_bars.png'%self.FOUT
        npl.savefig(fp)


if __name__ == "__main__":
    unittest.main()
