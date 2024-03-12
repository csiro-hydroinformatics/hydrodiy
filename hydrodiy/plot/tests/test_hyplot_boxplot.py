from pathlib import Path
import math
import pytest

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hydrodiy.plot.boxplot import Boxplot, BoxplotError

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"
FIMG.mkdir(exist_ok=True)

nval = 200
DATA = pd.DataFrame({
    "data1":np.random.normal(size=nval),
    "data2":np.random.normal(size=nval),
    "cat":np.random.randint(0, 5, size=nval)
})


def test_draw():
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.draw(ax=ax)
    bx.show_count()
    bx.show_count(ypos=0.975)
    fig.savefig(FIMG /"bx01_draw.png")


def test_draw_offset():
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.draw(ax=ax, xoffset=0.2)
    fig.savefig(FIMG /"bx01_draw_offset.png")


def test_draw_gca():
    plt.close("all")
    bx = Boxplot(data=DATA)
    bx.draw()
    plt.savefig(FIMG /"bx01_draw_gca.png")


def test_error():
    """ Test boxplot error """
    fig, ax = plt.subplots()

    msg = "Failed"
    with pytest.raises(BoxplotError, match=msg):
        data = [["a", "b", "c"], ["c", "d", "e"]]
        bx = Boxplot(data=data)

    msg = "Failed"
    with pytest.raises(BoxplotError, match=msg):
        data = np.random.uniform(0, 1, size=(10, 4))
        by = np.arange(data.shape[0]) % 2
        bx = Boxplot(data=data, by=by)


def test_draw_short():
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA[:5])
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx02_short.png")


def test_draw_props():
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.median.linecolor = "green"
    bx.box.linewidth = 5
    bx.minmax.show_line = True
    bx.minmax.marker = "*"
    bx.minmax.markersize = 20
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx03_props.png")


def test_by():
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA["data1"], by=DATA["cat"])
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx04_by.png")


def test_by_missing():
    fig, ax = plt.subplots()
    cat = pd.cut(DATA["cat"], range(-4, 5))
    bx = Boxplot(data=DATA["data1"], by=cat)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx10_by_missing1.png")


def test_by_missing2():
    df = pd.read_csv(FTEST / "boxplot_test_data.csv")
    cats = list(np.arange(0.8, 3.8, 0.2)) + [30]
    by = pd.cut(df["cat_value"], cats)

    fig, ax = plt.subplots()
    bx = Boxplot(data=df["value"], by=by)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx11_by_missing2.png")


def test_numpy():
    fig, ax = plt.subplots()
    data = np.random.uniform(0, 10, size=(1000, 6))
    bx = Boxplot(data=data)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx05_numpy.png")


def test_log():
    fig, ax = plt.subplots()
    data = DATA**2
    bx = Boxplot(data=data)
    bx.draw(ax=ax, logscale=True)
    bx.show_count()
    fig.savefig(FIMG /"bx06_log.png")


def test_width_by_count():
    fig, ax = plt.subplots()
    cat = DATA["cat"].copy()
    cat.loc[cat<3] = 0
    bx = Boxplot(data=DATA["data1"], by=cat,
                            width_from_count=True)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx07_width_count.png")


def test_coverage():
    plt.close("all")
    fig, axs = plt.subplots(ncols=2)

    bx1 = Boxplot(data=DATA)
    bx1.draw(ax=axs[0])
    axs[0].set_title("standard coverage 50/90")

    bx2 = Boxplot(data=DATA, box_coverage=40, whiskers_coverage=50)
    bx2.draw(ax=axs[1])
    axs[0].set_title("modified coverage 40/50")

    fig.savefig(FIMG /"bx08_coverage.png")


def test_coverage_by():
    plt.close("all")
    fig, ax = plt.subplots()
    cat = DATA["cat"].copy()
    cat.loc[cat<3] = 0
    bx = Boxplot(data=DATA["data1"], by=cat,
                whiskers_coverage=60)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx09_coverage_by.png")


def test_item_change():
    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.median.textformat = "%0.4f"
    bx.box.textformat = "%0.4f"
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx12_item_change.png")


def test_center():
    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.median.va = "bottom"
    bx.median.ha = "center"
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx13_center.png")

    msg = "Expected value in"
    with pytest.raises(BoxplotError, match=msg):
        bx.median.va = "left"


def test_nan():
    df = DATA.copy()
    df.loc[:, "data2"] = np.nan
    fig, ax = plt.subplots()
    bx = Boxplot(data=df)
    bx.draw(ax=ax)
    bx.show_count()
    fig.savefig(FIMG /"bx14_nan.png")


def test_narrow():
    nval = 200
    nvar = 50
    df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
            columns = ["data{0}".format(i) for i in range(nvar)])

    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(style="narrow", data=df)
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx15_narrow.png")


def test_showtext():
    nval = 200
    nvar = 5
    df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
            columns = ["data{0}".format(i) for i in range(nvar)])

    plt.close("all")
    fig, axs = plt.subplots(ncols=2)
    bx = Boxplot(data=df)
    bx.draw(ax=axs[0])

    bx = Boxplot(data=df, show_text=False)
    bx.draw(ax=axs[1])

    fig.savefig(FIMG /"bx16_showtext.png")


def test_center_text():
    nval = 200
    nvar = 5
    df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
            columns = ["data{0}".format(i) for i in range(nvar)])

    plt.close("all")
    fig, axs = plt.subplots(ncols=2)
    bx = Boxplot(data=df)
    bx.draw(ax=axs[0])

    bx = Boxplot(data=df, center_text=True)
    bx.draw(ax=axs[1])

    fig.savefig(FIMG /"bx17_center_text.png")


def test_number_format():
    nval = 200
    nvar = 5
    df = pd.DataFrame(np.random.normal(size=(nval, nvar)), \
            columns = ["data{0}".format(i) for i in range(nvar)])

    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=df, number_format="3.3e")
    bx.draw(ax=ax)
    fig.savefig(FIMG /"bx18_number_format.png")


def test_change_elements():
    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA, show_text=True)
    bx.draw(ax=ax)

    line = bx.elements["data2"]["median-line"]
    line.set_color("orange")
    line.set_solid_capstyle("round")
    line.set_linewidth(8)

    line = bx.elements["data1"]["top-cap1"]
    line.set_color("green")
    line.set_solid_capstyle("round")
    line.set_linewidth(6)

    txt = bx.elements["cat"]["median-text"]
    txt.set_weight("bold")
    txt.set_color("purple")

    fig.savefig(FIMG /"bx19_change_elements.png")


def test_set_ylim():
    plt.close("all")
    fig, axs = plt.subplots(ncols=2)
    ax = axs[0]
    bx = Boxplot(data=DATA)
    bx.draw(ax=ax)
    _, y1 = ax.get_ylim()
    bx.set_ylim((1, y1))

    ax = axs[1]
    bx.draw(ax=ax)
    bx.set_ylim((1, y1), False)

    fig.savefig(FIMG /"bx20_set_ylim.png")


def test_set_color():
    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA)
    bx.draw(ax=ax)
    bx.set_color(".*a2$", "tab:red")

    fig.savefig(FIMG /"bx21_set_color.png")

    plt.close("all")
    fig, ax = plt.subplots()
    bx = Boxplot(data=DATA, style="narrow")
    bx.draw(ax=ax)
    bx.set_color(".*a2$", "tab:red")

    fig.savefig(FIMG /"bx22_set_color_narrow.png")

