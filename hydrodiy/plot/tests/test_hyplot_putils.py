import os
import unittest

import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

from hydrodiy.plot import putils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.test = os.path.dirname(source_file)
        self.fimg = os.path.join(self.test, 'images')
        try:
            os.mkdir(self.fimg)
        except:
            pass

        # Reset matplotlib defaults
        mpl.rcdefaults()
        plt.close('all')


    def test_color_sets(self):
        ''' Test color sets '''
        for colname in ['slide_background', 'badgood', 'terciles', \
                            'cblind', 'safe']:
            cols = getattr(putils, 'COLORS_{}'.format(colname.upper()))
            if isinstance(cols, str):
                cols = [cols]

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], color='none')

            ncols = len(cols)
            for icol, col in enumerate(cols):
                r = Rectangle((icol/ncols, 0), \
                                1./ncols, 1., facecolor=col)
                ax.add_patch(r)

            fig.set_size_inches((12, 5))
            fp = os.path.join(self.fimg, 'colorset_{}.png'.format(colname))
            fig.savefig(fp)


    def test_colors2cmap(self):
        ''' Test conversion between color sets and color maps '''
        colors = {1:'#004C99', 0:'#FF9933', 0.3:'#FF99FF'}
        cmap = putils.colors2cmap(colors)

        x = np.arange(1, 257).reshape((1,256))
        fig, ax = plt.subplots()
        ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)
        fp = os.path.join(self.fimg, 'cmap.png')
        fig.savefig(fp)


    def test_grayscale(self):
        ''' Test conversion between color sets and color maps '''
        colors = {1:'#004C99', 0:'#FF9933', 0.3:'#FF99FF'}
        cmap = putils.colors2cmap(colors)

        grayscale = putils.cmap2grayscale(cmap)

        x = np.arange(1, 257).reshape((1,256))
        fig, axs = plt.subplots(ncols=2)
        axs[0].pcolor(x, cmap=cmap, vmin=1, vmax=256)
        axs[1].pcolor(x, cmap=grayscale, vmin=1, vmax=256)
        fp = os.path.join(self.fimg, 'grayscale.png')
        fig.savefig(fp)


    def test_cmap2colors(self):
        ''' Test conversion between color sets and color maps '''
        colors = putils.cmap2colors(ncols=10, cmap='Reds')
        self.assertTrue(len(colors) == 10)

        cmap = cm.get_cmap('Reds')
        colors = putils.cmap2colors(ncols=10, cmap=cmap)
        self.assertTrue(len(colors) == 10)

        for cmap in ['safe', 'PiYG']:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], color='none')

            ncols = 10
            cols = putils.cmap2colors(ncols, cmap)

            for icol, col in enumerate(cols):
                r = Rectangle((icol/ncols, 0), \
                                1./ncols, 1., facecolor=col)

                ax.add_patch(r)

            fig.set_size_inches((12, 5))
            fp = os.path.join(self.fimg, 'cmap2colors_{}.png'.format(cmap))
            fig.savefig(fp)



    def test_line(self):
        ''' Test line '''
        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        putils.line(ax, 0, 1, 0, 1, '-')
        putils.line(ax, 1, 0, 0, 0, '--')
        putils.line(ax, 1, 0.4, 0, 0, ':')
        putils.line(ax, 1, 0.2, 1., 2, '-.')

        fp = os.path.join(self.fimg, 'lines.png')
        fig.savefig(fp)


    def test_line_dates(self):
        ''' Test lines with dates in x axis '''
        fig, ax = plt.subplots()

        nval = 100
        x = pd.date_range('2001-01-01', periods=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        x0 = x[nval//2]
        putils.line(ax, 0, 1, x0, 1, '-')
        putils.line(ax, 1, 0, x0, 0, '--')
        putils.line(ax, 1, 0.4, x0, 0, ':')

        fp = os.path.join(self.fimg, 'lines_date.png')
        fig.savefig(fp)


    def test_equation(self):
        ''' Test equations '''
        tex = r'\begin{equation} y = ax+b \end{equation}'
        fp = os.path.join(self.fimg, 'equations1.png')
        try:
            putils.equation(tex, fp)
        except (FileNotFoundError, RuntimeError) as err:
            message = 'Cannot process tex command {0}'.format(tex)
            print(message)
            self.skipTest(message)

        tex = r'\begin{equation} y = \frac{\int_0^{+\infty}'+\
                            ' x\ \exp(-\\alpha x)}{\pi} \end{equation}'
        fp = os.path.join(self.fimg, 'equations2.png')
        try:
            putils.equation(tex, fp)
        except (FileNotFoundError, RuntimeError) as err:
            message = 'Cannot process tex command {0}'.format(tex)
            print(message)
            self.skipTest(message)

        tex = r'\begin{eqnarray} y & = & ax+b \\ z & = & \zeta \end{eqnarray}'
        fp = os.path.join(self.fimg, 'equations3.png')
        try:
            putils.equation(tex, fp)
        except (FileNotFoundError, RuntimeError) as err:
            message = 'Cannot process tex command {0}'.format(tex)
            print(message)
            self.skipTest(message)

        tex = r'\begin{equation} y = \begin{bmatrix} 1 & 0 & 0 \\ ' +\
            r'0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix} \end{equation}'
        fp = os.path.join(self.fimg, 'equations4.png')
        try:
            putils.equation(tex, fp)
        except (FileNotFoundError, RuntimeError) as err:
            message = 'Cannot process tex command {0}'.format(tex)
            print(message)
            self.skipTest(message)


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

        putils.set_mpl()
        fp = os.path.join(self.fimg, 'set_mpl1.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(color_theme='white')
        fp = os.path.join(self.fimg, 'set_mpl2.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(font_size=25)
        fp = os.path.join(self.fimg, 'set_mpl3.png')
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(usetex=True)
        fp = os.path.join(self.fimg, 'set_mpl4.png')
        try:
            plot(fp, True)
        except (FileNotFoundError, RuntimeError) as err:
            message = 'Cannot process set_mpl, error = {0}'.format(str(err))
            print(message)
            self.skipTest(message)

        mpl.rcdefaults()


    def test_kde(self):
        ''' Test kde generation '''
        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xx, yy, zz = putils.kde(xy)

        plt.close('all')
        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap='Blues')
        ax.contour(cont, colors='grey')
        ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, mfc='grey', mec='none')
        fp = os.path.join(self.fimg, 'kde.png')
        fig.savefig(fp)


    def test_kde_ties(self):
        ''' Test kde generation with ties '''
        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xy[:200, 0] = 1

        xx, yy, zz = putils.kde(xy)

        plt.close('all')
        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap='Blues')
        ax.contour(cont, colors='grey')
        ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, mfc='grey', mec='none')
        fp = os.path.join(self.fimg, 'kde_ties.png')
        fig.savefig(fp)


    def test_ellipse(self):
        ''' Test ellipse plot '''
        mu = [1, 2]
        fig, axs = plt.subplots(ncols=2)

        for irho, rho in enumerate([-0.9, 0.9]):
            cov = [[1, rho], [rho, 1]]
            xy = np.random.multivariate_normal(mu, cov, size=1000)

            ax = axs[irho]
            ax.plot(xy[:, 0], xy[:, 1], '.', alpha=0.2, \
                    mfc='grey', mec='none')

            colors = putils.cmap2colors(10, 'Reds')

            for i, pvalue in enumerate([0.5, 0.8, 0.9, 0.95, 0.99]):
                el = putils.cov_ellipse(mu, cov, pvalue, facecolor='none', \
                            edgecolor=colors[i])
                ax.add_patch(el)

        fp = os.path.join(self.fimg, 'ellipse.png')
        fig.savefig(fp)


    def test_qqplot(self):
        ''' Test qq plot '''
        putils.set_mpl()
        x = np.random.normal(size=200)

        plt.close('all')
        fig, ax = plt.subplots()
        putils.qqplot(ax, x)
        fp = os.path.join(self.fimg, 'qpplot1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        putils.qqplot(ax, x, True)
        ax.legend(loc=2)
        fp = os.path.join(self.fimg, 'qpplot2.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        xc = np.maximum(x, 1)
        putils.qqplot(ax, xc, True, 1)
        ax.legend(loc=2)
        fp = os.path.join(self.fimg, 'qpplot3.png')
        fig.savefig(fp)

        mpl.rcdefaults()


    def test_xdate_monthly(self):
        ''' Test formatting xaxis with monthly dates '''
        x = np.random.normal(size=200)
        dt = pd.date_range('1990-01-01', periods=len(x))
        dt = dt.to_pydatetime()

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax)
        fp = os.path.join(self.fimg, 'xdate_monthly1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, '3M')
        fp = os.path.join(self.fimg, 'xdate_monthly2.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'M', [2, 4, 5])
        fp = os.path.join(self.fimg, 'xdate_monthly3.png')
        fig.savefig(fp)


    def test_xdate_daily(self):
        ''' Test formatting xaxis with daily dates '''
        x = np.random.normal(size=200)
        dt = pd.date_range('1990-01-01', periods=len(x))
        dt = dt.to_pydatetime()

        plt.close('all')

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'D', by=[1, 15], format='%d\n%b\n%y')
        fp = os.path.join(self.fimg, 'xdate_daily.png')
        fig.savefig(fp)


    def test_xdate_yearly(self):
        ''' Test formatting xaxis with yearly dates '''
        x = np.random.normal(size=2000)
        dt = pd.date_range('1990-01-01', periods=len(x))
        dt = dt.to_pydatetime()

        plt.close('all')

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, 'Y', by=[1], format='%b\n%y')
        fp = os.path.join(self.fimg, 'xdate_yearl1.png')
        fig.savefig(fp)

        fig, ax = plt.subplots()
        ax.plot(dt, x)
        putils.xdate(ax, '3Y', by=[7], format='%b\n%y')
        fp = os.path.join(self.fimg, 'xdate_yearl2.png')
        fig.savefig(fp)


    def test_xdate_error(self):
        ''' Test xdate error with wrong x axis data '''
        x = np.random.normal(size=2000)
        dt = pd.date_range('1990-01-01', periods=len(x))

        plt.close('all')
        fig, ax = plt.subplots()

        try:
            ax.plot(dt, x)
            xticks = ax.get_xticks()
            putils.xdate(ax, 'Y', by=[1], format='%b\n%y')
            fp = os.path.join(self.fimg, 'xdate_yearl1.png')
            fig.savefig(fp)
        except ValueError as err:
            self.assertTrue(str(err).startswith('xaxis does not seem'))
        else:
            if np.any(xticks > 1e7):
                raise ValueError('Problem with error handling'+\
                                        ' (python 3 only)')


    def test_cmap_accentuate(self):
        ''' Test accentuation of cmap '''
        fig, axs = plt.subplots(nrows=3)

        u = np.linspace(0, 1, 250)
        zz = np.repeat(u[None, :], 10, axis=0)
        params = [1, 2.5, 4]
        levels = np.linspace(0, 1, 100)

        for iax, ax in enumerate(axs):
            cmap = putils.cmap_accentuate('RdBu', params[iax])
            ax.contourf(zz, levels=levels, vmin=0., vmax=1., cmap=cmap)

        fp = os.path.join(self.fimg, 'cmap_accentuate.png')
        fig.savefig(fp)


    def test_cmap_neutral(self):
        ''' Test accentuation of cmap '''
        u = np.linspace(0, 1, 250)
        zz = np.repeat(u[None, :], 10, axis=0)
        widths = [0.01, 0.05, 0.1]
        color = 'green'
        levels = np.linspace(0, 1, 100)

        fig, axs = plt.subplots(nrows=3)
        for iax, ax in enumerate(axs):
            cmap = putils.cmap_neutral('RdBu', neutral_color='green', \
                                            band_width=widths[iax])
            ax.contourf(zz, levels=levels, vmin=0., vmax=1., cmap=cmap)

        fp = os.path.join(self.fimg, 'cmap_neutral.png')
        fig.savefig(fp)


    def test_ecdfplot(self):
        ''' Test ecdf plots '''
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ['Var{}'.format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df)
        for nm in lines:
            lines[nm].set_linestyle(':')

        ax.legend(loc=2)
        fp = os.path.join(self.fimg, 'ecdfplot.png')
        fig.savefig(fp)


    def test_ecdfplot_nans(self):
        ''' Test ecdf plots with nan '''
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ['Var{}'.format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)
        df.loc[:800, 'Var1'] = np.nan
        df.loc[:990, 'Var2'] = np.nan

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df, label_stat='nunique', \
                                    label_stat_format='0.0f')
        ax.legend(loc=2)
        fp = os.path.join(self.fimg, 'ecdfplot_nan.png')
        fig.savefig(fp)


    def test_ecdfplot_labels(self):
        ''' Test ecdf plots with mean in labels'''
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ['Var{}'.format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df, 'std', '0.3f')

        ax.legend(loc=2)
        fp = os.path.join(self.fimg, 'ecdfplot_labels.png')
        fig.savefig(fp)


    def test_scattercat(self):
        ''' Test categorical scatter plot '''
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        fig, ax = plt.subplots()
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                alpha=0.6)
        ax.legend(loc=2, title='categories')
        fp = os.path.join(self.fimg, 'scattercat.png')
        fig.savefig(fp)


    def test_scattercat_nocmap(self):
        ''' Test categorical scatter plot with no cmap'''
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        fig, ax = plt.subplots()
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                cmap=None)
        ax.legend(loc=2, title='categories')
        fp = os.path.join(self.fimg, 'scattercat_nocmap.png')
        fig.savefig(fp)


    def test_scattercat_cat(self):
        ''' Test categorical scatter plot using categorical data '''
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        z = pd.Categorical(['oui' if zz > 0.7 else 'non' for zz in z])

        # Plot categorical data
        fig, axs = plt.subplots(ncols=2)
        ax = axs[0]
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                alpha=0.6)
        ax.legend(loc=2, title='categories')

        # Plot categorical data extracted from a dataframe
        ax = axs[1]
        df = pd.DataFrame({'z': z, 'a': np.nan})
        plotted, cats = putils.scattercat(ax, x, y, df.loc[:, 'z'], 5, \
                                markersizemin=5, markersizemax=12, \
                                alpha=0.6)
        ax.legend(loc=2, title='categories')

        fp = os.path.join(self.fimg, 'scattercat_cat.png')
        fig.savefig(fp)


    def test_interpolate_color(self):
        ''' Test color interpolation '''
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], '-', color='none')

        ncols = 20
        between = ['orange', 'blue']
        cols = [putils.interpolate_color('g', a, between) for a \
                    in np.linspace(0, 1, ncols)]

        for icol, col in enumerate(cols):
            r = Rectangle((icol/ncols, icol/ncols), \
                                1./ncols, 1./ncols, facecolor=col)
            ax.add_patch(r)

        fp = os.path.join(self.fimg, 'interpolate.png')
        fig.savefig(fp)


    def test_bivarnplot(self):
        ''' Test categorical scatter plot '''
        mean = [0, 0]
        cov = [[1, 0.7], [0.7, 1]]
        xy = np.random.multivariate_normal(mean, cov, size=100)
        fig, ax = plt.subplots()
        putils.bivarnplot(ax, xy)
        fp = os.path.join(self.fimg, 'bivarnplot.png')
        fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
