import os
import unittest

import numpy as np
import pandas as pd

from hydrodiy.gis import gutils

class GutilsTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> GutilsTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_xy2kml(self):
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


if __name__ == "__main__":
    unittest.main()

