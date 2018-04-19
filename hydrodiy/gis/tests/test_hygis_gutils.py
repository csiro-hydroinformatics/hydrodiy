import os
import unittest
import json

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

        fkml = '{0}/test5.kml'.format(self.ftest)
        gutils.xy2kml(x, y, fkml, siteid=siteid, icon='caution', scale=3)


    def test_georef(self):
        ''' Test the georef function with canberra '''
        try:
            lon, lat, xlim, ylim, info = gutils.georef(\
                            'Canberra ACT 2601, Australia')
        except:
            return

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
            out = gutils.georef('bidule, australia')
        except ValueError as err:
            self.assertTrue(str(err) == 'No results')
        else:
            raise ValueError('Problem with error handling')


    def test_georef_error_repeated(self):
        ''' Test repeated georef query '''

        try:
            for i in range(100):
                out = gutils.georef(\
                        'Canberra ACT 2601, Australia')
        except ValueError as err:
            self.assertTrue(str(err) in ['No results', 'Info is None'])
        else:
            raise ValueError('Problem with error handling')


if __name__ == "__main__":
    unittest.main()

