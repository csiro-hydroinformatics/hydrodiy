import os
import math
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydrodiy.plot.boxplot import Boxplot

class BoxplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BoxplotTestCase (hyplot)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

        nval = 200
        self.data = pd.DataFrame({
            'data1':np.random.normal(size=nval),
            'data2':np.random.normal(size=nval),
            'cat':np.random.randint(0, 5, size=nval)
        })


    def test_draw(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx1.png'))


    def test_draw_short(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data[:5])
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx2.png'))


    def test_draw_props(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data)
        bx['median'] = {'linecolor':'green'}
        bx['box'] = {'linewidth':5}
        bx['minmax'] = {'showline':True, 'marker':'*'}

        try:
            bx['med'] = {'linecolor':'green'}
        except ValueError as err:
            self.assertTrue(str(err).startswith('Cannot set property'))

        try:
            bx['median'] = {'bidule':3}
        except ValueError as err:
            self.assertTrue(str(err).startswith('Cannot set value'))

        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx3.png'))


    def test_by(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data['data1'], by=self.data['cat'])
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx4.png'))


    def test_numpy(self):
        fig, ax = plt.subplots()
        data = np.random.uniform(0, 10, size=(1000, 6))
        bx = Boxplot(ax=ax, data=data)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx5.png'))


    def test_log(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data+3)
        bx.draw(logscale=True)
        fig.savefig(os.path.join(self.FOUT, 'bx6.png'))



if __name__ == "__main__":
    unittest.main()
