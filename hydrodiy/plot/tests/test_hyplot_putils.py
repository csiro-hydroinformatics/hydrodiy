import os
import unittest

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from hydrodiy.plot import putils
from hydrodiy.gis import oz

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_col2cmap(self):

        colors = {1:'#004C99', 0:'#FF9933', 0.3:'#FF99FF'}
        cmap = putils.col2cmap(colors)

        x = np.arange(1, 257).reshape((1,256))
        fig, ax = plt.subplots()
        ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)
        fp = os.path.join(self.ftest, 'cmap.png')
        fig.savefig(fp)


    def test_line(self):

        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        putils.line(ax, 0, 1, 0, 1, '-')
        putils.line(ax, 1, 0, 0, 0, '--')
        putils.line(ax, 1, 0.4, 0, 0, ':')
        putils.line(ax, 1, 0.2, 1., 2, '-.')

        fp = os.path.join(self.ftest, 'lines.png')
        fig.savefig(fp)


    def test_line_dates(self):

        fig, ax = plt.subplots()

        nval = 100
        x = pd.date_range('2001-01-01', periods=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        x0 = x[nval/2]
        putils.line(ax, 0, 1, x0, 1, '-')
        putils.line(ax, 1, 0, x0, 0, '--')
        putils.line(ax, 1, 0.4, x0, 0, ':')

        fp = os.path.join(self.ftest, 'lines_date.png')
        fig.savefig(fp)


    def test_equation(self):

        tex = r'\begin{equation} y = ax+b \end{equation}'
        fp = os.path.join(self.ftest, 'equations1.png')
        putils.equation(tex, fp)


        tex = r'\begin{equation} y = \frac{\int_0^{+\infty} x\ \exp(-\alpha x)}{\pi} \end{equation}'
        fp = os.path.join(self.ftest, 'equations2.png')
        putils.equation(tex, fp)


        tex = r'\begin{eqnarray} y & = & ax+b \\ z & = & \zeta \end{eqnarray}'
        fp = os.path.join(self.ftest, 'equations3.png')
        putils.equation(tex, fp)


    def test_set_mpl(self):

        def plot(fp):
            fig, ax = plt.subplots()
            nval = 100
            x = np.arange(nval)
            y1 = np.random.normal(scale=2, size=nval)
            ax.plot(x, y1, 'o-', label='data1')
            y2 = np.random.normal(scale=2, size=nval)
            ax.plot(x, y2, 'o-', label='data2')
            leg = ax.legend()
            ax.set_title('Title')
            ax.set_xlabel('X label')
            ax.set_ylabel('Y label')
            fig.savefig(fp)

        putils.set_mpl()
        fp = os.path.join(self.ftest, 'set_mpl1.png')
        plot(fp)

        putils.set_mpl(color_theme='white')
        fp = os.path.join(self.ftest, 'set_mpl2.png')
        plot(fp)



    def test_kde(self):

        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xx, yy, zz = putils.kde(xy)

        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap='Blues')
        ax.contour(cont, colors='grey')
        ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, mfc='grey', mec='none')
        fp = os.path.join(self.ftest, 'kde.png')
        fig.savefig(fp)



if __name__ == "__main__":
    unittest.main()
