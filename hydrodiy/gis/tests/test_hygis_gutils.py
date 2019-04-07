import os
import unittest
import json

import numpy as np
import pandas as pd

from hydrodiy.gis import gutils
from hydrodiy.gis.gutils import HAS_C_GIS_MODULE

class GutilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GutilsTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.triangle = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]])


    def test_xy2kml(self):
        ''' Test conversion of points to kml format '''
        npt = 5
        x = np.linspace(145, 150, npt)
        y = np.linspace(-35, -30, npt)
        z = np.linspace(0, 100, npt)
        siteid = ['test'] * npt
        label = ['label'] * npt

        fkml = '{0}/test1.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml)

        fkml = '{0}/test2.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml, z=z)

        fkml = '{0}/test3.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml, siteid=siteid)

        fkml = '{0}/test4.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml, siteid=siteid, label=label)

        fkml = '{0}/test5.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml, siteid=siteid, icon='caution', scale=3)


    def test_georef(self):
        ''' Test the georef function with canberra '''
        try:
            lon, lat, xlim, ylim, info = gutils.georef(\
                            'Canberra ACT 2601, Australia')
        except:
            self.skipTest('Failure of georef')

        fj = os.path.join(self.ftest, 'canberra.json')
        with open(fj, 'r') as fo:
            info_e = json.load(fo)
        info_e['url'] = info['url']

        self.assertEqual(info, info_e)
        self.assertTrue(np.allclose([lon, lat], [149.1300092, -35.2809368]))
        self.assertTrue(np.allclose(xlim, (149.1207312, 149.1376675)))
        self.assertTrue(np.allclose(ylim, (-35.2873252, -35.2752841)))


    def test_georef_error(self):
        ''' Test error for georef '''
        try:
            out = gutils.georef('zzz_xwyxzz')
        except ValueError as err:
            self.assertTrue(str(err) in ['No results', 'Info is None'])
        else:
            raise ValueError('Problem with error handling')


    def test_point_inside_triangle(self):
        ''' Test points are inside a triangle '''
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
        inside = gutils.points_inside_polygon(points, self.triangle)
        expected = np.array([True, False, True])
        self.assertTrue(np.array_equal(inside, expected))

        points = np.array([[-0.2, -0.2], [0.2, 0.2], [1.0, 1.0],
                    [1e-30, 1e-30], [10., 10.]])
        inside = gutils.points_inside_polygon(points, self.triangle)
        expected = np.array([True, True, False, True, False])
        self.assertTrue(np.array_equal(inside, expected))


    def test_point_inside_polygon(self):
        ''' Test points are inside a polygon '''
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        # Additional data to test points in polygon algorithm
        fp = os.path.join(self.ftest, 'polygon.csv')
        xy = np.loadtxt(fp, delimiter=",")

        # Define grid
        xlim = xy[:, 0].min(), xy[:, 0].max()
        ylim = xy[:, 1].min(), xy[:, 1].max()
        x = np.linspace(*xlim, 30)
        y = np.linspace(*ylim, 30)
        xx, yy = np.meshgrid(x, y)

        # Compute inside/outside
        points = np.column_stack([xx.flat, yy.flat])
        inside = gutils.points_inside_polygon(points, xy)

        fp = os.path.join(self.ftest, 'polygon_inside.csv')
        expected = np.loadtxt(fp, delimiter=",").astype(bool)
        self.assertTrue(np.array_equal(inside, expected))


if __name__ == "__main__":
    unittest.main()

