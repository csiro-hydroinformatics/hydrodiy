import os
import unittest

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.plot import putils
from hydrodiy.gis import oz

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_col2cmap(self):
        ''' Test conversion between color sets and color maps '''

        mpl.rcdefaults()
        colors = {1:'#004C99', 0:'#FF9933', 0.3:'#FF99FF'}
        cmap = putils.col2cmap(colors)

        x = np.arange(1, 257).reshape((1,256))
        fig, ax = plt.subplots()
        ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)
        fp = os.path.join(self.ftest, 'cmap.png')
        fig.savefig(fp)


    def test_line(self):
        ''' Test line '''

        mpl.rcdefaults()
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
        ''' Test lines with dates in x axis '''

        mpl.rcdefaults()
        fig, ax = plt.subplots()

        nval = 100
        x = pd.date_range('2001-01-01', periods=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        x0 = x[nval//2]
        putils.line(ax, 0, 1, x0, 1, '-')
        putils.line(ax, 1, 0, x0, 0, '--')
        putils.line(ax, 1, 0.4, x0, 0, ':')

        fp = os.path.join(self.ftest, 'lines_date.png')
        fig.savefig(fp)


    def test_equation(self):
        ''' Test equations '''

        mpl.rcdefaults()

        plt.close('all')
        mpl.rcdefaults()
        tex = r'\begin{equation} y = ax+b \end{equation}'
        fp = os.path.join(self.ftest, 'equations1.png')
        putils.equation(tex, fp)

        tex = r'\begin{equation} y = \frac{\int_0^{+\infty} x\ \exp(-\alpha x)}{\pi} \end{equation}'
        fp = os.path.join(self.ftest, 'equations2.png')
        putils.equation(tex, fp)

        tex = r'\begin{eqnarray} y & = & ax+b \\ z & = & \zeta \end{eqnarray}'
        fp = os.path.join(self.ftest, 'equations3.png')
        putils.equation(tex, fp)

        tex = r'\begin{equation} y = \begin{bmatrix} 1 & 0 & 0 \\ ' +\
            r'0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} \end{equation}'
        fp = os.path.join(self.ftest, 'equations4.png')
        putils.equation(tex, fp, height=500)


    def test_set_mpl(self):
        ''' Test set mpl '''

        def plot(fp, usetex=False):
            fig, ax = plt.subplots()
            nval = 100
            x = np.arange(nval)
            y1 = np.random.normal(scale=2, size=nval)
            ax.plot(x, y1, 'o-', label='x')
            y2 = np.random.normal(scale=2, size=nval)

            label = 'y'
            if usetex:
                label='$\displaystyle \sum_1^\infty x^i$'
            ax.plot(x, y2, 'o-', label=label)
            leg = ax.legend()
            ax.set_title('Title')
            ax.set_xlabel('X label')
            ax.set_ylabel('Y label')
            fig.savefig(fp)

        plt.close('all')
        mpl.rcdefaults()
        putils.set_mpl()
        fp = os.path.join(self.ftest, 'set_mpl1.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(color_theme='white')
        fp = os.path.join(self.ftest, 'set_mpl2.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(font_size=25)
        fp = os.path.join(self.ftest, 'set_mpl3.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(usetex=True)
        fp = os.path.join(self.ftest, 'set_mpl4.png')
        plot(fp, True)
        mpl.rcdefaults()

    def test_kde(self):
        ''' Test kde generation '''

        mpl.rcdefaults()

        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xx, yy, zz = putils.kde(xy)

        plt.close('all')
        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap='Blues')
        ax.contour(cont, colors='grey')
        ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, mfc='grey', mec='none')
        fp = os.path.join(self.ftest, 'kde.png')
        fig.savefig(fp)


    def test_ellipse(self):
        ''' Test ellipse plot '''

        mpl.rcdefaults()

        mu = [1, 2]
        cov = [[1, 0.9], [0.9, 1]]
        xy = np.random.multivariate_normal(mu, cov, size=1000)

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, mfc='grey', mec='none')

        colors = putils.get_colors(10, 'Reds')

        for i, pvalue in enumerate([0.5, 0.8, 0.9, 0.95, 0.99]):
            el = putils.cov_ellipse(mu, cov, pvalue, facecolor='none', \
                        edgecolor=colors[i])
            ax.add_patch(el)

        fp = os.path.join(self.ftest, 'ellipse.png')
        fig.savefig(fp)


    def test_qqplot(self):
        ''' Test qq plot '''

        mpl.rcdefaults()

        putils.set_mpl()
        x = np.random.normal(size=200)

        plt.close('all')
        fig, ax = plt.subplots()
        putils.qqplot(ax, x)
        fp = os.path.join(self.ftest, 'qpplot1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        putils.qqplot(ax, x, True)
        ax.legend(loc=2)
        fp = os.path.join(self.ftest, 'qpplot2.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        xc = np.maximum(x, 1)
        putils.qqplot(ax, xc, True, 1)
        ax.legend(loc=2)
        fp = os.path.join(self.ftest, 'qpplot3.png')
        fig.savefig(fp)

        mpl.rcdefaults()


    def test_xdate_monthly(self):
        ''' Test formatting xaxis with monthly dates '''

        #putils.set_mpl(font_size=10)
        mpl.rcdefaults()

        x = np.random.normal(size=200)
        dt = pd.date_range('1990-01-01', periods=len(x))

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax)
        fp = os.path.join(self.ftest, 'xdate_monthly1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, '3M')
        fp = os.path.join(self.ftest, 'xdate_monthly2.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'M', [2, 4, 5])
        fp = os.path.join(self.ftest, 'xdate_monthly3.png')
        fig.savefig(fp)


    def test_xdate_daily(self):
        ''' Test formatting xaxis with daily dates '''

        #putils.set_mpl(font_size=10)
        mpl.rcdefaults()

        x = np.random.normal(size=200)
        dt = pd.date_range('1990-01-01', periods=len(x))

        plt.close('all')

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'D', by=[1, 15], format='%d\n%b\n%y')
        fp = os.path.join(self.ftest, 'xdate_daily.png')
        fig.savefig(fp)


    def test_xdate_yearly(self):
        ''' Test formatting xaxis with yearly dates '''

        #putils.set_mpl(font_size=10)
        mpl.rcdefaults()

        x = np.random.normal(size=2000)
        dt = pd.date_range('1990-01-01', periods=len(x))

        plt.close('all')

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'Y', by=[1], format='%b\n%y')
        fp = os.path.join(self.ftest, 'xdate_yearl1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, '3Y', by=[7], format='%b\n%y')
        fp = os.path.join(self.ftest, 'xdate_yearl2.png')
        fig.savefig(fp)


    def test_get_fig_axs(self):
        ''' Test generation of fig and axs '''

        fig, axs = putils.get_fig_axs()
        self.assertTrue(isinstance(axs, mpl.axes.Axes))

        fig, axs = putils.get_fig_axs(nrows=2, ncols=2)
        self.assertTrue(isinstance(axs, np.ndarray))
        self.assertTrue(np.allclose(axs.shape, (4, )))

        fig, axs = putils.get_fig_axs(nrows=2, ncols=2, ravel=False)
        self.assertTrue(isinstance(axs, np.ndarray))
        self.assertTrue(np.allclose(axs.shape, (2, 2)))

if __name__ == "__main__":
    unittest.main()
