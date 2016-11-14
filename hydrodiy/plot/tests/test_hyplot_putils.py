import os
import unittest

import pandas as pd
import numpy as np
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


    def test_set_spines(self):

        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        putils.set_spines(ax, ['right', 'top'], visible=False)
        putils.set_spines(ax, ['left', 'bottom'], color='red', style=':')

        fp = os.path.join(self.ftest, 'spines.png')
        fig.savefig(fp)


    def test_set_legend(self):

        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)
        ax.plot(x, y, label='data')
        leg = ax.legend()
        putils.set_legend(leg, textcolor='green', framealpha=0.5)

        fp = os.path.join(self.ftest, 'legend.png')
        fig.savefig(fp)

    def test_set_mpl(self):

        putils.set_mpl()
        fig, ax = plt.subplots()
        nval = 100
        x = np.arange(nval)
        y1 = np.random.normal(scale=2, size=nval)
        ax.plot(x, y1, 'o-', label='data1')
        y2 = np.random.normal(scale=2, size=nval)
        ax.plot(x, y2, 'o-', label='data2')
        leg = ax.legend()
        fp = os.path.join(self.ftest, 'set_mpl1.png')
        fig.savefig(fp)


        putils.set_mpl(True)
        fig, ax = plt.subplots()
        nval = 100
        x = np.arange(nval)
        y1 = np.random.normal(scale=2, size=nval)
        ax.plot(x, y1, 'o-', label='data1')
        y2 = np.random.normal(scale=2, size=nval)
        ax.plot(x, y2, 'o-', label='data2')
        leg = ax.legend()
        fp = os.path.join(self.ftest, 'set_mpl2.png')
        fig.savefig(fp)



if __name__ == "__main__":
    unittest.main()
