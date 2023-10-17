from pathlib import Path
import os
import math
import pytest

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hydrodiy.plot.violinplot import Violin, ViolinplotError

FHERE = Path(__file__).resolve().parent
FIMG = FHERE / "images"
FIMG.mkdir(exist_ok=True)

NVAL = 200
DATA1 = pd.DataFrame({
    "data1":np.random.normal(size=NVAL),
    "data2":np.random.normal(size=NVAL)+np.random.normal(size=NVAL)*3+0.5,
    "data3":np.exp(np.random.normal(size=NVAL)*1-1)
})


DATA2 = pd.DataFrame({
    "data1":1+np.exp(np.random.normal(size=NVAL)*3-1),
    "data2":2+np.exp(np.random.normal(size=NVAL)*2-1)
})

def test_violin_init(allclose):
    vl = Violin(data=DATA1)

    assert vl.data.shape == DATA1.shape
    ncol = DATA1.shape[1]
    assert len(vl.stat_center_low) == ncol
    assert len(vl.stat_center_high) == ncol
    assert len(vl.stat_extremes_low) == ncol
    assert len(vl.stat_extremes_high) == ncol
    assert len(vl.stat_median) == ncol
    assert vl.kde_x.shape[1] == ncol
    assert vl.kde_y.shape[1] == ncol
    assert allclose(vl.kde_y.max(), 1)


def test_violin_draw():
    plt.close("all")
    vl = Violin(data=DATA1)
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot.png"
    fig.savefig(fp)


def test_violin_brightening():
    plt.close("all")
    vl = Violin(data=DATA1, bfl=-0.1, bfsl=-0.2)
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_bright.png"
    fig.savefig(fp)


def test_violin_notext():
    plt.close("all")
    vl = Violin(data=DATA1, st=False)
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_notext.png"
    fig.savefig(fp)


def test_violin_colors():
    plt.close("all")
    vl = Violin(data=DATA1, crm="red", cro="pink")
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_colors1.png"
    fig.savefig(fp)

    fig, ax = plt.subplots()
    vl.show_text = False
    vl.col_ref_median = "none"
    vl.col_ref_others = "tab:orange"
    vl.reset_items()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_colors2.png"
    fig.savefig(fp)


def test_violin_draw_extremes():
    plt.close("all")
    vl = Violin(data=DATA2)
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_extremes.png"
    fig.savefig(fp)


def test_violin_log():
    plt.close("all")
    vl = Violin(data=DATA2)
    fig, ax = plt.subplots()
    vl.draw(ax=ax)
    ax.set_yscale("log")
    fp = FIMG / "violin_log.png"
    fig.savefig(fp)


