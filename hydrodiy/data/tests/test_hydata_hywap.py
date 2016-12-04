import os, re
import unittest
import itertools

import numpy as np
from  datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.data import hywap
from hydrodiy.gis.grid import AWAP_GRID


class HyWapTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyWapTestCase (hydata)')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        fawap = os.path.join(self.ftest, 'awap')
        if not os.path.exists(fawap):
            os.mkdir(fawap)

        self.fawap = fawap


    def test_get_data(self):
        dt = datetime(2015, 2, 1)
        vn = hywap.VARIABLES
        ts = hywap.TIMESTEPS
        plt.close('all')

        for varname, timestep in itertools.product(vn.keys(), ts):

            for v in vn[varname]:
                vartype = v['type']

                grd = hywap.get_data(varname, vartype,
                                            timestep, dt)

                st = iutils.dict2str({'varname':varname, \
                    'timestep': timestep, \
                    'vartype': vartype} )
                fg = os.path.join(self.fawap, '{0}.bil'.format(st))
                grd.save(fg)

                print(varname, vartype)

                # Check  grid except for solar
                if varname != 'solar':
                    for attr in ['nrows', 'ncols', 'cellsize']:
                        self.assertEqual(getattr(grd, attr), \
                            getattr(AWAP_GRID, attr))

                fig, ax = plt.subplots()
                grd.plot(ax, cmap='Blues')
                ax.set_title('{0} - {1} - {2}'.format(varname, vartype, dt))
                fig.savefig(re.sub('bil', 'png', fg))


if __name__ == "__main__":
    unittest.main()
