import os
import unittest

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from matplotlib import pyplot as plt
from hygis.oz import Oz
from hyplot import putils

class OzTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> OzTestCase')
        FTEST, ozfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_oz0(self):

        plt.close('all')

        om = Oz()
        om.drawcoastoz()
        om.drawstates()
      
        fp = '%s/oz_plot0.png'%self.FOUT
        plt.savefig(fp)

    def test_oz1(self):

        plt.close('all')
        fig, ax = plt.subplots()

        om = Oz(ax)
        om.drawcoast()
        om.drawstates()

        npt = 100
        x = np.random.normal(loc=133, scale=20, size=npt)
        y = np.random.normal(loc=-25, scale=20, size=npt)
        om.plot(x, y, 'ro')
       
        putils.footer(fig) 
        fp = '%s/oz_plot1.png'%self.FOUT
        fig.savefig(fp)

    def test_oz2(self):

        plt.close('all')

        om = Oz()
        om.drawrelief()
        om.drawcoastoz(color='blue')
        om.drawstates(color='red', linestyle='--')
      
        fp = '%s/oz_plot2.png'%self.FOUT
        plt.savefig(fp)

    def test_oz3(self):

        plt.close('all')

        om = Oz()
        om.drawcoast()
        om.drawstates()

        om.set_lim([135, 157],[-24, -39])
      
        fp = '%s/oz_plot3.png'%self.FOUT
        plt.savefig(fp)

    def test_oz4(self):

        # Create shapefile
        npt = 100
        x = np.random.normal(loc=133, scale=20, size=npt)
        dx = np.ones_like(x) * 1
        y = np.random.normal(loc=-25, scale=20, size=npt)
        dy = np.ones_like(x) * 1

        df = gpd.GeoDataFrame({'geometry':
                [Polygon([(xx,yy), (xx+dxx, yy), 
                    (xx+dxx, yy+dyy), (xx, yy+dyy)]) 
                    for xx, dxx, yy, dyy in zip(x,dx,y,dy)]})

        fshp = '%s/test' % self.FOUT
        df.to_file('%s.shp' % fshp, driver='ESRI Shapefile')

        plt.close('all')

        om = Oz()
        om.drawcoast()
        om.drawstates(linestyle='--')

        om.drawpolygons(fshp, edgecolor='red', facecolor='none')

      
        fp = '%s/oz_plot4.png'%self.FOUT
        plt.savefig(fp)


    def test_oz5(self):

        plt.close('all')

        om = Oz()
        om.drawcoastoz('k-')
        om.drawstates()

        fp = '%s/oz_plot5.png'%self.FOUT
        plt.savefig(fp)


if __name__ == "__main__":
    unittest.main()

