import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

from hydrodiy.plot import putils


FHERE = Path(__file__).resolve().parent
FIMG = FHERE / "images"
FIMG.mkdir(exist_ok=True)


def test_color_sets():
    """ Test color sets """
    for colname in ["badgood", "terciles", "cblind", "safe", \
                    "core", "primary", "secondary", "nature"]:
        cols = getattr(putils, "COLORS_{}".format(colname.upper()))
        if isinstance(cols, str):
            cols = [cols]
        elif isinstance(cols, dict):
            cols = [col for cn, col in cols.items()]

        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color="none")

        ncols = len(cols)
        for icol, col in enumerate(cols):
            r = Rectangle((icol/ncols, 0), \
                            1./ncols, 1., facecolor=col)
            ax.add_patch(r)

        fig.set_size_inches((12, 5))
        fp = FIMG / f"colorset_{colname}.png"
        fig.savefig(fp)


def test_colors2cmap():
    """ Test conversion between color sets and color maps """
    colors = {1:"#004C99", 0:"#FF9933", 0.3:"#FF99FF"}
    cmap = putils.colors2cmap(colors)

    x = np.arange(1, 257).reshape((1,256))
    fig, ax = plt.subplots()
    ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)
    fp = FIMG / "cmap.png"
    fig.savefig(fp)


def test_cmap2colors():
    """ Test conversion between color sets and color maps """
    colors = putils.cmap2colors(ncols=10, cmap="Reds")
    assert len(colors) == 10

    cmap = cm.get_cmap("Reds")
    colors = putils.cmap2colors(ncols=10, cmap=cmap)
    assert len(colors) == 10

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
        fp = FIMG / f"cmap2colors_{cmap}.png"
        fig.savefig(fp)



def test_line():
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

    fp = os.path.join(FIMG, "lines.png")
    fp = FIMG / "lines.png"
    fig.savefig(fp)


def test_line_dates():
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

    fp = FIMG / "lines_date.png"
    fig.savefig(fp)


def test_set_mpl():
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
    fp = FIMG / "set_mpl1.png"
    plot(fp)

    mpl.rcdefaults()
    putils.set_mpl(color_theme="white")
    fp = FIMG / "set_mpl2.png"
    plot(fp)

    mpl.rcdefaults()
    putils.set_mpl(font_size=25)
    fp = FIMG / "set_mpl3.png"
    plot(fp)

    mpl.rcdefaults()
    putils.set_mpl(color_cycle=putils.COLORS_NATURE)
    fp = FIMG / "set_mpl4.png"
    plot(fp)

    mpl.rcdefaults()
    putils.set_mpl(usetex=True)
    fp = FIMG / "set_mpl5.png"
    try:
        plot(fp, True)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        message = "Cannot process set_mpl, error = {0}".format(str(err))
        print(message)

    mpl.rcdefaults()


def test_kde():
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
    fp = FIMG / "kde.png"
    fig.savefig(fp)


def test_kde_ties():
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
    fp = FIMG / "kde_ties.png"
    fig.savefig(fp)


def test_ellipse():
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

    fp = FIMG / "ellipse.png"
    fig.savefig(fp)


def test_qqplot():
    """ Test qq plot """
    putils.set_mpl()
    x = np.random.normal(size=200)

    plt.close("all")
    fig, ax = plt.subplots()
    putils.qqplot(ax, x)
    fp = FIMG / "qqplot1.png"
    fig.savefig(fp)

    fig, ax = plt.subplots()
    putils.qqplot(ax, x, True)
    ax.legend(loc=2)
    fp = FIMG / "qqplot2.png"
    fig.savefig(fp)

    fig, ax = plt.subplots()
    xc = np.maximum(x, 1)
    putils.qqplot(ax, xc, True, 1)
    ax.legend(loc=2)
    fp = FIMG / "qqplot3.png"
    fig.savefig(fp)

    mpl.rcdefaults()


def test_ecdfplot():
    """ Test ecdf plots """
    df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
    cc = ["Var{}".format(i+1) for i in range(4)]
    df = pd.DataFrame(df, columns=cc)

    fig, ax = plt.subplots()
    lines = putils.ecdfplot(ax, df)
    for nm in lines:
        lines[nm].set_linestyle(":")

    ax.legend(loc=2)
    fp = FIMG / "ecdf_plot.png"
    fig.savefig(fp)


def test_ecdfplot_nans():
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
    fp = FIMG / "ecdf_plot_nan.png"
    fig.savefig(fp)


def test_ecdfplot_labels():
    """ Test ecdf plots with mean in labels"""
    df = np.random.normal(size=(1000, 4)) + np.arange(4)[None, :]
    cc = ["Var{}".format(i+1) for i in range(4)]
    df = pd.DataFrame(df, columns=cc)

    fig, ax = plt.subplots()
    lines = putils.ecdfplot(ax, df, "std", "0.3f")

    ax.legend(loc=2)
    fp = FIMG / "ecdf_plot_labels.png"
    fig.savefig(fp)


def test_scattercat():
    """ Test categorical scatter plot """
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    fig, ax = plt.subplots()
    plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            markersizes=np.linspace(30, 70, 5), \
                            alphas=0.6)
    ax.legend(loc=2, title="categories")
    fp = FIMG / "scattercat.png"
    fig.savefig(fp)


def test_scattercat_counts():
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    fig, ax = plt.subplots()
    plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            markersizes=np.linspace(30, 70, 5), \
                            alphas=0.6, scl=True)
    ax.legend(loc=2, title="categories")
    fp = FIMG / "scattercat_counts.png"
    fig.savefig(fp)


def test_scattercat_nocmap():
    """ Test categorical scatter plot with no cmap"""
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    fig, ax = plt.subplots()
    plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            markersizes=np.linspace(30, 70, 5), \
                            cmap=None)
    ax.legend(loc=2, title="categories")
    fp = FIMG / "scattercat_nocmap.png"
    fig.savefig(fp)


def test_scattercat_markers():
    """ Test categorical scatter plot with different markers"""
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    fig, ax = plt.subplots()
    markers = ["o", "s", "d", "^", "v"]
    plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            markers=markers, \
                            show_extremes_in_legend=False, \
                            cmap=None)
    ax.legend(loc=2, title="categories")
    fp = FIMG / "scattercat_markers.png"
    fig.savefig(fp)


def test_scattercat_edgecolors():
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    fig, ax = plt.subplots()

    cols = ["red", "orange", "white", "lightblue", "blue"]
    lotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            ec=cols, \
                            ms=[30, 80], \
                            cmap="PiYG")
    ax.legend(loc=2, title="categories")
    fp = FIMG / "scattercat_edgecolors.png"
    fig.savefig(fp)


def test_scattercat_cat():
    """ Test categorical scatter plot using categorical data """
    x, y, z = np.random.uniform(0, 1, size=(100, 3)).T
    z = pd.Categorical(["oui" if zz > 0.7 else "non" for zz in z])

    # Plot categorical data
    fig, axs = plt.subplots(ncols=2)
    ax = axs[0]
    plotted, cats = putils.scattercat(ax, x, y, z, 5, \
                            alphas=0.6)
    ax.legend(loc=2, title="categories")
    ax.set_title("z is Series")

    # Plot categorical data extracted from a dataframe
    ax = axs[1]
    df = pd.DataFrame({"z": z, "a": np.nan})
    plotted, cats = putils.scattercat(ax, x, y, df.loc[:, "z"], 5, \
                            alphas=0.6)
    ax.legend(loc=2, title="categories")
    ax.set_title("z is from Data frame")

    fp = FIMG / "scattercat_cat.png"
    fig.savefig(fp)



def test_bivarnplot(allclose):
    """ Test categorical scatter plot """
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    nsmp = 100000
    xy = np.random.multivariate_normal(mean, cov, size=nsmp)
    fig, ax = plt.subplots()

    unorm, rho, eta, rho_p, rho_m = putils.bivarnplot(ax, xy)

    fp = FIMG / "bivarnplot.png"
    fig.savefig(fp)

    assert allclose(rho, 0.7, 1e-2)
    assert allclose(rho_p, eta, 1e-2)
    assert allclose(rho_m, eta, 1e-2)


def test_waterbalplot():
    """ Test categorical scatter plot """
    plt.close("all")
    fig, ax = plt.subplots()
    tm_line = putils.waterbalplot(ax)
    tm_line.set_linewidth(5)
    tm_line.set_color("tab:red")
    fp = FIMG / "waterbalplot.png"
    fig.savefig(fp)


def test_blackwhite():
    plt.close("all")
    fig, ax = plt.subplots()
    x = np.random.uniform(0, 1, size=(50, 3))
    ax.scatter(x[:, 0], x[:, 1], c=x[:,2])
    fp = FIMG / "blackwhite.png"
    fig.savefig(fp)
    putils.blackwhite(fp)


def test_darken_lightend():
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

    fp = FIMG / "darken_lighten.png"
    fig.savefig(fp)
    putils.blackwhite(fp)


