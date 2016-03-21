import os
import unittest
import numpy as np
from hydrodiy.gis import gutils

class PolygonTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> PolygonTestCase')
        self.triangle = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]])

        FTEST, ozfile = os.path.split(__file__)
        p_file = os.path.join(FTEST, 'polygon.csv')
        self.polygon = np.loadtxt(p_file, delimiter=",")

    def test_triangle(self):
        points = np.array([[0.2, 0.2], [1.0, 1.0], [-0.2, -0.2]])
        inside = gutils.points_inside_polygon(points, self.triangle)
        expected = np.array([True, False, True])
        self.assertTrue(np.array_equal(inside, expected))

    def test_polygon(self):
        points = np.array([[-0.2, -0.2], [0.2, 0.2], [1.0, 1.0],
                    [1e-30, 1e-30], [10., 10.]])
        inside = gutils.points_inside_polygon(points, self.triangle)
        expected = np.array([True, True, False, True, False])
        self.assertTrue(np.array_equal(inside, expected))

if __name__ == "__main__":
    unittest.main()
