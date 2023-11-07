from pathlib import Path
import pytest


import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from hydrodiy.plot import simplot
from hydrodiy.plot import putils

# Reset matplotlib to default
mpl.rcdefaults()

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"
FIMG.mkdir(exist_ok=True)

def test_sim_daily():
    dt = pd.date_range("1970-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var", sim_name="bidule")

    sm.add_sim(sim2, name="truc")
    axb, axa, axfd, axfdl, axs, axf = sm.draw()

    fp = FIMG / "simplot_daily.png"
    sm.savefig(fp)


def test_sim_daily_samefloodyscale():
    dt = pd.date_range("1970-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var", sim_name="bidule", samefloodyscale=True)

    sm.add_sim(sim2, name="truc")
    axb, axa, axfd, axfdl, axs, axf = sm.draw()
    ylims = []
    for ax in axf:
        ylims.append(ax.get_ylim())
    ylims = np.array(ylims)
    assert np.all(np.std(ylims, 0) < 1e-10)

    fp = FIMG / "simplot_daily_samefloodyscale.png"
    sm.savefig(fp)


def test_nfloods():
    dt = pd.date_range("2000-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim1, "Var", sim_name="bidule", nfloods=10)
    sm.draw()

    fp = FIMG / "simplot_nfloods.png"
    sm.savefig(fp)


def test_nfloods():
    dt = pd.date_range("2000-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim1 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim2 = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim1, "Var", sim_name="bidule")
    sm.draw()

    fp = FIMG / "simplot_add_sim.png"
    sm.savefig(fp)


def test_sim_monthly():
    dt = pd.date_range("2000-01-01", "2015-12-01", freq="MS")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var")
    sm.draw()

    fp = FIMG / "simplot_monthly.png"
    sm.savefig(fp)


def test_axis():
    dt = pd.date_range("1970-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var", sim_name="bidule")

    axb, axa, axfd, axfdl, axs, axf = sm.draw()

    axb.set_title("T1")
    axa.set_title("T2")
    axfd.set_title("T3")
    axfdl.set_title("T4")
    axs.set_title("T5")
    for ax in axf:
        ax.set_ylim([0, 1])

    fp = FIMG / "simplot_axis.png"
    sm.savefig(fp)


def test_options_fdc_zoom():
    dt = pd.date_range("1970-01-01", "2015-12-01")
    nval = len(dt)

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var", sim_name="bidule", \
                fdc_zoom_xlim=[0., 0.2], \
                fdc_zoom_ylog=False)

    axb, axa, axfd, axfdl, axs, axf = sm.draw()

    fp = FIMG / "simplot_fdc_zoom.png"
    sm.savefig(fp)


def test_color_scheme():
    dt = pd.date_range("1970-01-01", "2015-12-01")
    nval = len(dt)

    sch = simplot.COLOR_SCHEME
    cols = putils.cmap2colors(3, "Spectral")
    simplot.COLOR_SCHEME = cols

    obs = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)
    sim = pd.Series(np.exp(np.random.normal(size=nval)), index=dt)

    plt.close("all")
    sm = simplot.Simplot(obs, sim, "Var", sim_name="bidule")
    axb, axa, axfd, axfdl, axs, axf = sm.draw()

    fp = FIMG / "simplot_colors.png"
    sm.savefig(fp)

    simplot.COLOR_SCHEME = sch


