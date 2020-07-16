import os
import unittest

from hydrodiy.gis.oz import HAS_BASEMAP, Oz, REGIONS, \
                                HAS_PYSHP, ozlayer, HYGisOzError

from hydrodiy.plot import putils

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

HAS_PYPROJ = False
try:
    import pyproj
    HAS_PYPROJ = True
except (ImportError, FileNotFoundError) as err:
    pass



class OzTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> OzTestCase')
        if not HAS_BASEMAP:
            self.skipTest('Import error')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)


    def test_region(self):
        ''' Test set region boundaries '''
        for reg in REGIONS:
            plt.close('all')
            fig, ax = plt.subplots()
            om = Oz(ax=ax)
            om.drawcoast()
            om.drawstates(linewidth=2)
            om.drawdrainage(linewidth=1, linestyle='-', color='grey')
            om.set_lim_region(reg)
            fp = os.path.join(self.fimg, 'oz_region_{0}.png'.format(reg))
            plt.savefig(fp)


    def test_set_lim(self):
        plt.close('all')
        fig, ax = plt.subplots()
        om = Oz(ax=ax)
        om.drawcoast()
        om.drawstates()
        om.set_lim([130, 140], [-20, -10])
        fp = os.path.join(self.fimg, 'oz_set_lim.png')
        plt.savefig(fp)


    def test_oz_coast_hires(self):
        plt.close('all')
        om = Oz()
        om.drawcoast(hires=True)
        fp = os.path.join(self.fimg, 'oz_coast_hires.png')
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

        fp = os.path.join(self.fimg, 'oz_plot.png')
        fig.savefig(fp)


    def test_oz_relief(self):
        plt.close('all')

        om = Oz()

        # Skip if decoder jpeg is not available
        try:
            om.drawrelief()
        except IOError as err:
            self.assertTrue(str(err) == 'decoder jpeg not available')
        except MemoryError:
            import warnings
            warnings.warn('test_oz_relief no run due to memory error')
            return

        om.drawcoast(edgecolor='blue')
        om.drawstates(color='red', linestyle='--')

        fp = os.path.join(self.fimg, 'oz_relief.png')
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
        om.drawcoast(linestyle=':', linewidth=3, \
            edgecolor='r', facecolor='b', alpha=0.5)
        fp = os.path.join(self.fimg, 'oz_coast_style.png')
        plt.savefig(fp)


    def test_oz_drainage_style(self):
        plt.close('all')
        om = Oz()
        om.drawcoast()
        om.drawdrainage(linestyle='--', color='r')
        fp = os.path.join(self.fimg, 'oz_drainage_style.png')
        plt.savefig(fp)



class OzLayerTestCase(unittest.TestCase):


    def setUp(self):
        print('\t=> OzLayerTestCase')
        if not HAS_PYSHP:
            self.skipTest('Import error')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)


    def test_ozmap_layers(self):
        ''' Test plotting oz map layers '''
        for layer in  ['ozcoast10m', 'ozcoast50m', 'drainage', 'states50m']:
            plt.close('all')
            fig, ax = plt.subplots()
            lines = ozlayer(ax, layer, color='k', lw=0.5)

            fp = os.path.join(self.fimg, 'ozlayer_{}.png'.format(layer))
            plt.savefig(fp)

            if layer == 'drainage':
                self.assertEqual(len(lines), 233)


    def test_ozmap_layers_filter_error(self):
        ''' Test plotting oz map layers using filter '''
        plt.close('all')
        fig, ax = plt.subplots()
        try:
            lines = ozlayer(ax, 'drainage',  \
                    filter_field='bidule', \
                    filter_regex='South East', \
                    color='k', lw=0.5)
        except HYGisOzError as err:
            self.assertTrue(str(err).startswith('Expected filter_field'))
            return


    def test_ozmap_layers_filter(self):
        ''' Test plotting oz map layers using filter '''
        plt.close('all')
        fig, ax = plt.subplots()
        lines = ozlayer(ax, 'drainage',  \
                    filter_field='Division', \
                    filter_regex='South East', \
                    color='k', lw=0.5)
        fp = os.path.join(self.fimg, 'ozlayer_drainage_filter.png')
        plt.savefig(fp)


    def test_ozmap_layers_proj(self):
        ''' Test plotting oz map layers using projection '''

        if not HAS_PYPROJ:
            self.skipTest('Missing pyproj module')
        proj = pyproj.Proj('+init=EPSG:3112')

        plt.close('all')
        fig, ax = plt.subplots()
        lines = ozlayer(ax, 'ozcoast50m',  proj=proj, \
                    color='k', lw=2)
        fp = os.path.join(self.fimg, 'ozlayer_drainage_proj.png')
        plt.savefig(fp)


if __name__ == "__main__":
    unittest.main()

