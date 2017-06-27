import os
import unittest

import numpy as np
import pandas as pd

# Check availability of shapely
has_shapely = False
try:
    from shapely.geometry import Polygon
    has_shapely = True
except ImportError:
    pass

from matplotlib import pyplot as plt
from hydrodiy.gis.oz import Oz, REGIONS
from hydrodiy.plot import putils

class OzTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> OzTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_set_lim_region(self):
        plt.close('all')
        fig, ax = plt.subplots()
        om = Oz(ax=ax)
        om.drawcoast()
        om.drawstates()
        for reg in REGIONS:
            om.set_lim_region(reg)

        # Add points
        xlim = [147.5, 155.]
        ylim = [-38.5, -29.9]
        x = np.random.uniform(xlim[0], xlim[1], 100)
        y = np.random.uniform(ylim[0], ylim[1], 100)
        om.plot(x, y, 'o')

        om.set_lim_region('COASTALNSW')

        fp = os.path.join(self.ftest, 'oz_region.png')
        plt.savefig(fp)


    def test_set_lim(self):
        plt.close('all')
        fig, ax = plt.subplots()
        om = Oz(ax=ax)
        om.drawcoast()
        om.drawstates()
        om.set_lim([130, 140], [-20, -10])
        fp = os.path.join(self.ftest, 'oz_set_lim.png')
        plt.savefig(fp)


    def test_oz_coast_hires(self):
        plt.close('all')
        om = Oz()
        om.drawcoast(hires=True)
        fp = os.path.join(self.ftest, 'oz_coast_hires.png')
        plt.savefig(fp)


    def test_oz(self):
        plt.close('all')
        fig, ax = plt.subplots()

        om = Oz(ax)
        om.drawcoast()
        om.drawstates()

        npt = 100
        x = np.random.normal(loc=133, scale=20, size=npt)
        y = np.random.normal(loc=-25, scale=20, size=npt)
        om.plot(x, y, 'ro')

        fp = os.path.join(self.ftest, 'oz_plot.png')
        fig.savefig(fp)


    def test_oz_relief(self):
        plt.close('all')

        om = Oz()

        # Skip if decoder jpeg is not available
        try:
            om.drawrelief()
        except IOError as err:
            self.assertTrue(str(err) == 'decoder jpeg not available')

        om.drawcoast(edgecolor='blue')
        om.drawstates(color='red', linestyle='--')

        fp = os.path.join(self.ftest, 'oz_relief.png')
        plt.savefig(fp)

    #def test_oz4(self):

    #    # Create shapefile
    #    npt = 100
    #    x = np.random.normal(loc=133, scale=20, size=npt)
    #    dx = np.ones_like(x) * 1
    #    y = np.random.normal(loc=-25, scale=20, size=npt)
    #    dy = np.ones_like(x) * 1

    #    df = gpd.GeoDataFrame({'geometry':
    #            [Polygon([(xx,yy), (xx+dxx, yy),
    #                (xx+dxx, yy+dyy), (xx, yy+dyy)])
    #                for xx, dxx, yy, dyy in zip(x,dx,y,dy)]})

    #    fshp = '%s/test' % self.ftest
    #    df.to_file('%s.shp' % fshp, driver='ESRI Shapefile')

    #    plt.close('all')

    #    om = Oz()
    #    om.drawcoast()
    #    om.drawstates(linestyle='--')

    #    om.drawpolygons(fshp, edgecolor='red', facecolor='none')


    #    fp = '%s/oz_4.png'%self.ftest
    #    plt.savefig(fp)


    def test_oz_coast_style(self):
        plt.close('all')
        om = Oz()
        om.drawcoast(edgecolor='r', facecolor='b', alpha=0.1)
        fp = os.path.join(self.ftest, 'oz_coast_style.png')
        plt.savefig(fp)


    def test_oz_drainage_style(self):
        plt.close('all')
        om = Oz()
        om.drawcoast()
        om.drawdrainage(linestyle='--', color='r')
        fp = os.path.join(self.ftest, 'oz_drainage_style.png')
        plt.savefig(fp)


if __name__ == "__main__":
    unittest.main()

