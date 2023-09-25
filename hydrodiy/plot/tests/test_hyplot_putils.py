import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

from hydrodiy.plot import putils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print("\t=> UtilsTestCase (hyplot)")
        source_file = Path(__file__).resolve()
        self.test = source_file.parent
        self.fimg = self.test / "images"
        self.fimg.mkdir(exist_ok=True)

        # Reset matplotlib defaults
        mpl.rcdefaults()
        plt.close("all")


    def test_color_sets(self):
        """ Test color sets """
        for colname in ["badgood", "terciles", "cblind", "safe", \
                        "core", "primary", "secondary"]:
            cols = getattr(putils, "COLORS_{}".format(colname.upper()))
            if isinstance(cols, str):
                cols = [cols]
            elif isinstance(cols, dict):
                cols = [col for cn, col in cols.items()]

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], color="none")

            ncols = len(cols)
            for icol, col in enumerate(cols):
                r = Rectangle((icol/ncols, 0), \
                                1./ncols, 1., facecolor=col)
                ax.add_patch(r)

            fig.set_size_inches((12, 5))
            fp = self.fimg / f"colorset_{colname}.png"
            fig.savefig(fp)


    def test_colors2cmap(self):
        """ Test conversion between color sets and color maps """
        colors = {1:"#004C99", 0:"#FF9933", 0.3:"#FF99FF"}
        cmap = putils.colors2cmap(colors)

        x = np.arange(1, 257).reshape((1,256))
        fig, ax = plt.subplots()
        ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)
        fp = self.fimg / "cmap.png"
        fig.savefig(fp)


    def test_cmap2colors(self):
        """ Test conversion between color sets and color maps """
        colors = putils.cmap2colors(ncols=10, cmap="Reds")
        self.assertTrue(len(colors) == 10)

        cmap = cm.get_cmap("Reds")
        colors = putils.cmap2colors(ncols=10, cmap=cmap)
        self.assertTrue(len(colors) == 10)

        for cmap in ["safe", "PiYG"]:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], color="none")

            ncols = 10
            cols = putils.cmap2colors(ncols, cmap)

            for icol, col in enumerate(cols):
                r = Rectangle((icol/ncols, 0), \
                                1./ncols, 1., facecolor=col)

                ax.add_patch(r)

            fig.set_size_inches((12, 5))
            fp = self.fimg / f"cmap2colors_{cmap}.png"
            fig.savefig(fp)



    def test_line(self):
        """ Test line """
        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        putils.line(ax, 0, 1, 0, 1, "-")
        putils.line(ax, 1, 0, 0, 0, "--")
        putils.line(ax, 1, 0.4, 0, 0, ":")
        putils.line(ax, 1, 0.2, 1., 2, "-.")

        fp = os.path.join(self.fimg, "lines.png")
        fp = self.fimg / "lines.png"
        fig.savefig(fp)


    def test_line_dates(self):
        """ Test lines with dates in x axis """
        fig, ax = plt.subplots()

        nval = 100
        x = pd.date_range("2001-01-01", periods=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        x0 = x[nval//2]
        putils.line(ax, 0, 1, x0, 1, "-")
        putils.line(ax, 1, 0, x0, 0, "--")
        putils.line(ax, 1, 0.4, x0, 0, ":")

        fp = self.fimg / "lines_date.png"
        fig.savefig(fp)


    def test_set_mpl(self):
        """ Test set mpl """

        def plot(fp, usetex=False):
            fig, ax = plt.subplots()
            nval = 100
            x = np.arange(nval)
            y1 = np.random.normal(scale=2, size=nval)
            ax.plot(x, y1, "o-", label="x")
            y2 = np.random.normal(scale=2, size=nval)

            label = "y"
            if usetex:
                label=r"$\displaystyle \sum_1^\infty x^i$"
            ax.plot(x, y2, "o-", label=label)
            leg = ax.legend()
            ax.set_title("Title")
            ax.set_xlabel("X label")
            ax.set_ylabel("Y label")
            fig.savefig(fp)

        putils.set_mpl()
        fp = self.fimg / "set_mpl1.png"
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(color_theme="white")
        fp = self.fimg / "set_mpl2.png"
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(font_size=25)
        fp = self.fimg / "set_mpl3.png"
        plot(fp)

        mpl.rcdefaults()
        putils.set_mpl(usetex=True)
        fp = self.fimg / "set_mpl4.png"
        try:
            plot(fp, True)
        except (FileNotFoundError, RuntimeError) as err:
            message = "Cannot process set_mpl, error = {0}".format(str(err))
            print(message)
            self.skipTest(message)

        mpl.rcdefaults()


    def test_kde(self):
        """ Test kde generation """
        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xx, yy, zz = putils.kde(xy)

        plt.close("all")
        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap="Blues")
        ax.contour(cont, colors="grey")
        ax.plot(xy[:, 0], xy[:, 1], ".", alpha=0.2, mfc="grey", mec="none")
        fp = self.fimg / "kde.png"
        fig.savefig(fp)


    def test_kde_ties(self):
        """ Test kde generation with ties """
        xy = np.random.multivariate_normal( \
            [1, 2], [[1, 0.9], [0.9, 1]], \
            size=1000)

        xy[:200, 0] = 1

        xx, yy, zz = putils.kde(xy)

        plt.close("all")
        fig, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz, cmap="Blues")
        ax.contour(cont, colors="grey")
        ax.plot(xy[:, 0], xy[:, 1], ".", alpha=0.2, mfc="grey", mec="none")
        fp = self.fimg / "kde_ties.png"
        fig.savefig(fp)


    def test_ellipse(self):
        """ Test ellipse plot """
        mu = [1, 2]
        fig, axs = plt.subplots(ncols=2)

        for irho, rho in enumerate([-0.9, 0.9]):
            cov = [[1, rho], [rho, 1]]
            xy = np.random.multivariate_normal(mu, cov, size=1000)

            ax = axs[irho]
            ax.plot(xy[:, 0], xy[:, 1], ".", alpha=0.2, \
                    mfc="grey", mec="none")

            colors = putils.cmap2colors(10, "Reds")

            for i, pvalue in enumerate([0.5, 0.8, 0.9, 0.95, 0.99]):
                el = putils.cov_ellipse(mu, cov, pvalue, facecolor="none", \
                            edgecolor=colors[i])
                ax.add_patch(el)

        fp = self.fimg / "ellipse.png"
        fig.savefig(fp)


    def test_qqplot(self):
        """ Test qq plot """
        putils.set_mpl()
        x = np.random.normal(size=200)

        plt.close("all")
        fig, ax = plt.subplots()
        putils.qqplot(ax, x)
        fp = self.fimg / "qqplot1.png"
        fig.savefig(fp)

        fig, ax = plt.subplots()
        putils.qqplot(ax, x, True)
        ax.legend(loc=2)
        fp = self.fimg / "qqplot2.png"
        fig.savefig(fp)

        fig, ax = plt.subplots()
        xc = np.maximum(x, 1)
        putils.qqplot(ax, xc, True, 1)
        ax.legend(loc=2)
        fp = self.fimg / "qqplot3.png"
        fig.savefig(fp)

        mpl.rcdefaults()


    def test_ecdfplot(self):
        """ Test ecdf plots """
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ["Var{}".format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df)
        for nm in lines:
            lines[nm].set_linestyle(":")

        ax.legend(loc=2)
        fp = self.fimg / "ecdf_plot.png"
        fig.savefig(fp)


    def test_ecdfplot_nans(self):
        """ Test ecdf plots with nan """
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ["Var{}".format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)
        df.loc[:800, "Var1"] = np.nan
        df.loc[:990, "Var2"] = np.nan

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df, label_stat="nunique", \
                                    label_stat_format="0.0f")
        ax.legend(loc=2)
        fp = self.fimg / "ecdf_plot_nan.png"
        fig.savefig(fp)


    def test_ecdfplot_labels(self):
        """ Test ecdf plots with mean in labels"""
        df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
        cc = ["Var{}".format(i+1) for i in range(4)]
        df = pd.DataFrame(df, columns=cc)

        fig, ax = plt.subplots()
        lines = putils.ecdfplot(ax, df, "std", "0.3f")

        ax.legend(loc=2)
        fp = self.fimg / "ecdf_plot_labels.png"
        fig.savefig(fp)


    def test_scattercat(self):
        """ Test categorical scatter plot """
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        fig, ax = plt.subplots()
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                alphas=0.6)
        ax.legend(loc=2, title="categories")
        fp = self.fimg / "scattercat.png"
        fig.savefig(fp)


    def test_scattercat_nocmap(self):
        """ Test categorical scatter plot with no cmap"""
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        fig, ax = plt.subplots()
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                cmap=None)
        ax.legend(loc=2, title="categories")
        fp = self.fimg / "scattercat_nocmap.png"
        fig.savefig(fp)


    def test_scattercat_cat(self):
        """ Test categorical scatter plot using categorical data """
        x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
        z = pd.Categorical(["oui" if zz > 0.7 else "non" for zz in z])

        # Plot categorical data
        fig, axs = plt.subplots(ncols=2)
        ax = axs[0]
        plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                                markersizemin=5, markersizemax=12, \
                                alphas=0.6)
        ax.legend(loc=2, title="categories")

        # Plot categorical data extracted from a dataframe
        ax = axs[1]
        df = pd.DataFrame({"z": z, "a": np.nan})
        plotted, cats = putils.scattercat(ax, x, y, df.loc[:, "z"], 5, \
                                markersizemin=5, markersizemax=12, \
                                alphas=0.6)
        ax.legend(loc=2, title="categories")

        fp = self.fimg / "scattercat_cat.png"
        fig.savefig(fp)



    def test_bivarnplot(self):
        """ Test categorical scatter plot """
        mean = [0, 0]
        cov = [[1, 0.7], [0.7, 1]]
        xy = np.random.multivariate_normal(mean, cov, size=100)
        fig, ax = plt.subplots()
        putils.bivarnplot(ax, xy)
        fp = self.fimg / "bivarnplot.png"
        fig.savefig(fp)


    def test_waterbalplot(self):
        """ Test categorical scatter plot """
        plt.close("all")
        fig, ax = plt.subplots()
        tm_line = putils.waterbalplot(ax)
        tm_line.set_linewidth(5)
        tm_line.set_color("tab:red")
        fp = self.fimg / "waterbalplot.png"
        fig.savefig(fp)


    def test_png_metadata(self):
        plt.close("all")
        fig, ax = plt.subplots()
        x = np.random.uniform(0, 1, size=50)
        ax.plot(x)
        fp = self.fimg / "metadata.png"
        fig.savefig(fp)

        meta = {\
            "source_file": Path(__file__).resolve(), \
            "bidule": "truc", \
            "xdata": ", ".join([f"{xx:0.2f}" for xx in x])
        }
        putils.add_metadata_to_png(fp, meta)

        meta2 = putils.read_metadata_from_png(fp)
        for k in meta2:
            if k in ["author", "time_created"]:
                continue
            assert meta2[k] == str(meta[k])


    def test_blackwhite(self):
        plt.close("all")
        fig, ax = plt.subplots()
        x = np.random.uniform(0, 1, size=(50, 3))
        ax.scatter(x[:, 0], x[:, 1], c=x[:,2])
        fp = self.fimg / "blackwhite.png"
        fig.savefig(fp)
        putils.blackwhite(fp)


    def test_darken_lightend(self):
        plt.close("all")
        fig, ax = plt.subplots()

        col_ref = "tab:purple"
        modifs = [-1, -0.5, 0, 0.5, 1]

        x = np.linspace(0, 1, 100)
        for im, m in enumerate(modifs):
            c = putils.darken_or_lighten(col_ref, m)
            lab = f"Modifier = {m}"
            ax.plot(x, (im+1)*x, color=c, lw=5, label=lab)

        ax.legend(loc=4)

        fp = self.fimg / "darken_lighten.png"
        fig.savefig(fp)
        putils.blackwhite(fp)



if __name__ == "__main__":
    unittest.main()
