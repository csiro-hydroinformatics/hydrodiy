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
        fig.savefig(os.path.join(self.FOUT, 'bx01_draw.png'))


    def test_draw_short(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data[:5])
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx02_short.png'))


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
        fig.savefig(os.path.join(self.FOUT, 'bx03_props.png'))


    def test_by(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data['data1'], by=self.data['cat'])
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx04_by.png'))


    def test_by_missing(self):
        fig, ax = plt.subplots()
        cat = pd.cut(self.data['cat'], range(-4, 5))
        bx = Boxplot(ax=ax, data=self.data['data1'], by=cat)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx10_by_missing1.png'))


    def test_by_missing2(self):
        df = pd.read_csv(os.path.join(self.FOUT, 'boxplot_test_data.csv'))
        cats = list(np.arange(0.8, 3.8, 0.2)) + [30]
        by = pd.cut(df['cat_value'], cats)

        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=df['value'], by=by)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx11_by_missing2.png'))

    def test_numpy(self):
        fig, ax = plt.subplots()
        data = np.random.uniform(0, 10, size=(1000, 6))
        bx = Boxplot(ax=ax, data=data)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx05_numpy.png'))


    def test_log(self):
        fig, ax = plt.subplots()
        data = self.data**2
        bx = Boxplot(ax=ax, data=data)
        bx['count'] = {'showtext':True}
        bx.draw(logscale=True)
        fig.savefig(os.path.join(self.FOUT, 'bx06_log.png'))


    def test_width_by_count(self):
        fig, ax = plt.subplots()
        cat = self.data['cat']
        cat.loc[cat<3] = 0
        bx = Boxplot(ax=ax, data=self.data['data1'], by=cat,
                                width_from_count=True)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx07_width_count.png'))


    def test_coverage(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data, whiskers_coverage=51)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx08_coverage.png'))


    def test_coverage_by(self):
        fig, ax = plt.subplots()
        cat = self.data['cat']
        cat.loc[cat<3] = 0
        bx = Boxplot(ax=ax, data=self.data['data1'], by=cat,
                    whiskers_coverage=60)
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx09_coverage_by.png'))


    def test_set_all(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data)
        bx.set_all('textformat', '%0.4f')
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx12_set_all.png'))


    def test_center(self):
        fig, ax = plt.subplots()
        bx = Boxplot(ax=ax, data=self.data)
        bx.set_all('ha', 'center')
        bx.set_all('va', 'bottom')
        bx.draw()
        fig.savefig(os.path.join(self.FOUT, 'bx13_center.png'))



if __name__ == "__main__":
    unittest.main()