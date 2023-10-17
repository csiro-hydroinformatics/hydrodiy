import os
import unittest
import json

import numpy as np
import pandas as pd

from hydrodiy.gis import gutils
from hydrodiy import has_c_module

class GutilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GutilsTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.triangle = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]])


    def test_point_inside_triangle(self):
        ''' Test points are inside a triangle '''
        if not has_c_module("gis", False):
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
        if not has_c_module("gis", False):
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


    def test_point_inside_polygon_memory_efficient(self):
        ''' Test points are inside a polygon without creating
            the inside vector
        '''
        if not has_c_module("gis", False):
            self.skipTest('Missing C module c_hydrodiy_gis')

        points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
        inside = np.zeros(3, dtype=np.int32)
        gutils.points_inside_polygon(points, self.triangle, \
                                inside=inside)
        expected = np.array([True, False, True])
        self.assertTrue(np.array_equal(inside, expected))


    def test_point_inside_polygon_memory_efficient_error(self):
        ''' Test points are inside a polygon without creating
            the inside vector - test for errors.
        '''
        if not has_c_module("gis", False):
            self.skipTest('Missing C module c_hydrodiy_gis')

        points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
        inside = np.zeros(2, dtype=np.int32)
        try:
            gutils.points_inside_polygon(points, self.triangle, \
                                inside=inside)
        except ValueError as err:
            self.assertTrue(str(err).startswith(\
                            'Expected inside of length'))
        else:
            raise ValueError('Problem in error handling')


        inside = np.zeros(3)
        try:
            gutils.points_inside_polygon(points, self.triangle, \
                                inside=inside)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected inside of dtype'))
        else:
            raise ValueError('Problem in error handling')

        inside = np.array(['']*3)
        try:
            gutils.points_inside_polygon(points, self.triangle, \
                                inside=inside)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected inside of dtype'))
        else:
            raise ValueError('Problem in error handling')


if __name__ == "__main__":
    unittest.main()

