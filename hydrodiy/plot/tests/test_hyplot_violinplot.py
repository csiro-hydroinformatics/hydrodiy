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
DATA = pd.DataFrame({
    "data1":np.random.normal(size=NVAL),
    "data2":np.random.normal(size=NVAL)+np.random.normal(size=NVAL)*3+0.5,
    "data3":np.exp(np.random.normal(size=NVAL)*1-1)
})


def test_violin_init(allclose):
    vl = Violin(data=DATA)

    assert vl.data.shape == DATA.shape
    ncol = DATA.shape[1]
    assert len(vl.stat_center_low) == ncol
    assert len(vl.stat_center_high) == ncol
    assert len(vl.stat_extremes_low) == ncol
    assert len(vl.stat_extremes_high) == ncol
    assert len(vl.stat_median) == ncol
    assert vl.kde_x.shape[1] == ncol
    assert vl.kde_y.shape[1] == ncol
    assert allclose(vl.kde_y.max(), 1)


def test_violin_draw():
    vl = Violin(data=DATA)

    fig, ax = plt.subplots()
    vl.draw(ax=ax)

    fp = FIMG / "violin_plot.png"
    fig.savefig(fp)


    fig, ax = plt.subplots()
    vl.col_ref_others = "tab:orange"
    vl.set_items()
    vl.draw(ax=ax)
    fp = FIMG / "violin_plot_colors.png"
    fig.savefig(fp)


