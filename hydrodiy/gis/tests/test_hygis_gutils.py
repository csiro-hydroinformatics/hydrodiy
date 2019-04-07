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

