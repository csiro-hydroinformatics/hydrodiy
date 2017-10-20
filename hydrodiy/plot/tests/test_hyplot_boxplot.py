import os
import math
import unittest

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.plot.boxplot import Boxplot

class BoxplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BoxplotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        nval = 200
        self.data = pd.DataFrame({
            'data1':np.random.normal(size=nval),
            'data2':np.random.normal(size=nval),
            'cat':np.random.randint(0, 5, size=nval)
        })


    def test_draw(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data)
        bx.draw(ax=ax)
        bx.show_count()
        bx.show_count(ypos=0.975)
        fig.savefig(os.path.join(self.ftest, 'bx01_draw.png'))


    def test_draw_offset(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data)
        bx.draw(ax=ax, xoffset=0.2)
        fig.savefig(os.path.join(self.ftest, 'bx01_draw_offset.png'))


    def test_draw_gca(self):
        plt.close('all')
        bx = Boxplot(data=self.data)
        bx.draw()
        plt.savefig(os.path.join(self.ftest, 'bx01_draw_gca.png'))


    def test_error(self):
        ''' Test boxplot error '''
        fig, ax = plt.subplots()

        try:
            data = [['a', 'b', 'c'], ['c', 'd', 'e']]
            bx = Boxplot(data=data)
        except Exception as err:
            self.assertTrue(str(err).startswith('Failed'))
        else:
            raise Exception('Problem with error handling')

        try:
            data = np.random.uniform(0, 1, size=(10, 4))
            by = np.arange(data.shape[0]) % 2
            bx = Boxplot(data=data, by=by)
        except Exception as err:
            self.assertTrue(str(err).startswith('Failed'))
        else:
            raise Exception('Problem with error handling')


    def test_draw_short(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data[:5])
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx02_short.png'))


    def test_draw_props(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data)
        bx.median.linecolor = 'green'
        bx.box.linewidth = 5
        bx.minmax.showline = True
        bx.minmax.marker = '*'
        bx.minmax.markersize = 20
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx03_props.png'))


    def test_by(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data['data1'], by=self.data['cat'])
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx04_by.png'))


    def test_by_missing(self):
        fig, ax = plt.subplots()
        cat = pd.cut(self.data['cat'], range(-4, 5))
        bx = Boxplot(data=self.data['data1'], by=cat)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx10_by_missing1.png'))


    def test_by_missing2(self):
        df = pd.read_csv(os.path.join(self.ftest, 'boxplot_test_data.csv'))
        cats = list(np.arange(0.8, 3.8, 0.2)) + [30]
        by = pd.cut(df['cat_value'], cats)

        fig, ax = plt.subplots()
        bx = Boxplot(data=df['value'], by=by)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx11_by_missing2.png'))


    def test_numpy(self):
        fig, ax = plt.subplots()
        data = np.random.uniform(0, 10, size=(1000, 6))
        bx = Boxplot(data=data)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx05_numpy.png'))


    def test_log(self):
        fig, ax = plt.subplots()
        data = self.data**2
        bx = Boxplot(data=data)
        bx.draw(ax=ax, logscale=True)
        bx.show_count()
        fig.savefig(os.path.join(self.ftest, 'bx06_log.png'))


    def test_width_by_count(self):
        fig, ax = plt.subplots()
        cat = self.data['cat'].copy()
        cat.loc[cat<3] = 0
        bx = Boxplot(data=self.data['data1'], by=cat,
                                width_from_count=True)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx07_width_count.png'))


    def test_coverage(self):
        fig, axs = plt.subplots(ncols=2)

        bx1 = Boxplot(data=self.data)
        bx1.draw(ax=axs[0])
        axs[0].set_title('standard coverage 50/90')

        bx2 = Boxplot(data=self.data, box_coverage=40, whiskers_coverage=50)
        bx2.draw(ax=axs[1])
        axs[0].set_title('modified coverage 40/50')

        fig.savefig(os.path.join(self.ftest, 'bx08_coverage.png'))


    def test_coverage_by(self):
        fig, ax = plt.subplots()
        cat = self.data['cat'].copy()
        cat.loc[cat<3] = 0
        bx = Boxplot(data=self.data['data1'], by=cat,
                    whiskers_coverage=60)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx09_coverage_by.png'))


    def test_item_change(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data)
        bx.median.textformat = '%0.4f'
        bx.box.textformat = '%0.4f'
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx12_item_change.png'))


    def test_center(self):
        fig, ax = plt.subplots()
        bx = Boxplot(data=self.data)
        bx.median.va = 'bottom'
        bx.median.ha = 'center'
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx13_center.png'))

        try:
            bx.median.va = 'left'
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected value in'))
        else:
            raise Exception('Problem with error handling')


    def test_nan(self):
        df = self.data
        df.loc[:, 'data2'] = np.nan
        fig, ax = plt.subplots()
        bx = Boxplot(data=df)
        bx.draw(ax=ax)
        bx.show_count()
        fig.savefig(os.path.join(self.ftest, 'bx14_nan.png'))


    def test_narrow(self):
        ''' Testing narrow style '''

        nval = 200
        nvar = 50
        df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
                columns = ['data{0}'.format(i) for i in range(nvar)])

        fig, ax = plt.subplots()
        bx = Boxplot(style='narrow', data=df)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx15_narrow.png'))


    def test_showtext(self):
        ''' Testing showtext '''

        nval = 200
        nvar = 5
        df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
                columns = ['data{0}'.format(i) for i in range(nvar)])

        fig, axs = plt.subplots(ncols=2)
        bx = Boxplot(data=df)
        bx.draw(ax=axs[0])

        bx = Boxplot(data=df, showtext=False)
        bx.draw(ax=axs[1])

        fig.savefig(os.path.join(self.ftest, 'bx16_showtext.png'))


    def test_centertext(self):
        ''' Testing centertext '''

        nval = 200
        nvar = 5
        df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
                columns = ['data{0}'.format(i) for i in range(nvar)])

        fig, axs = plt.subplots(ncols=2)
        bx = Boxplot(data=df)
        bx.draw(ax=axs[0])

        bx = Boxplot(data=df, centertext=True)
        bx.draw(ax=axs[1])

        fig.savefig(os.path.join(self.ftest, 'bx17_centertext.png'))


    def test_digitnumber(self):
        ''' Testing digitnumber '''

        nval = 200
        nvar = 5
        df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
                columns = ['data{0}'.format(i) for i in range(nvar)])

        fig, ax = plt.subplots()
        bx = Boxplot(data=df, digitnumber=4)
        bx.draw(ax=ax)
        fig.savefig(os.path.join(self.ftest, 'bx18_digitnumber.png'))




if __name__ == "__main__":
    unittest.main()
