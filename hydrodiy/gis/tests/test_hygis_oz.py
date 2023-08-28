import os
from pathlib import Path
import pytest

from hydrodiy.gis.oz import REGIONS, HAS_PYSHP, ozlayer, ozcities
from hydrodiy.plot import putils

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe

FHERE = Path(__file__).resolve().parent
FIMG = FHERE / "images"
FIMG.mkdir(exist_ok=True)

HAS_PYPROJ = False
try:
    import pyproj
    HAS_PYPROJ = True
except (ImportError, FileNotFoundError) as err:
    pass


def test_ozlayer():
    """ Test plotting oz map layers """
    layers = ["ozcoast10m", "ozcoast50m", "ozdrainage", \
                "ozstates50m", "ozbasins"]
    for layer in layers:
        plt.close("all")
        fig, ax = plt.subplots()
        lines = ozlayer(ax, layer, color="k", lw=0.5, fixed_lim=False)

        fp = FIMG / f"ozlayer_{layer}.png"
        plt.savefig(fp)

        if layer == "ozdrainage":
            assert len(lines) == 233


def test_ozlayer_filter_error():
    """ Test plotting oz map layers using filter """
    plt.close("all")
    fig, ax = plt.subplots()

    msg = "Expected filter_field"
    with pytest.raises(ValueError, match=msg):
        lines = ozlayer(ax, "ozdrainage",  \
                filter_field="bidule", \
                filter_regex="South East", \
                color="k", lw=0.5)


def test_ozlayer_filter():
    """ Test plotting oz map layers using filter """
    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozdrainage",  \
                filter_field="Division", \
                filter_regex="South East", \
                color="k", lw=0.5, fixed_lim=False)
    fp = FIMG / "ozlayer_drainage_filter.png"
    plt.savefig(fp)


def test_ozlayer_proj():
    """ Test plotting oz map layers using projection """

    if not HAS_PYPROJ:
        pytest.skip("Missing pyproj module")

    proj = pyproj.Proj("+init=EPSG:3112")

    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  proj=proj, \
                color="k", lw=2)
    fp = FIMG / "ozlayer_drainage_proj.png"
    plt.savefig(fp)


def test_ozcities():
    """ Test plotting oz cities """
    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  \
                color="k", lw=0.5, fixed_lim=False)

    kw = {
        "path_effects": [pe.withStroke(linewidth=3, foreground="w")], \
        "textcoords": "offset pixels",
        "ha": "center", \
        "xytext": (0, 8)
    }
    elems = ozcities(ax, text_kwargs=kw)

    fp = FIMG / "ozcities_kw.png"
    plt.savefig(fp)


def test_ozcities_filter():
    """ Test plotting oz cities using filter """
    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  \
                color="k", lw=0.5, fixed_lim=False)
    elems = ozcities(ax, filter_regex="Melbourne|Perth")

    fp = FIMG / "ozcities_filter.png"
    plt.savefig(fp)


def test_ozcities_options():
    """ Test plotting oz cities with optiopns """
    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  \
                color="k", lw=0.5, fixed_lim=False)
    elems = ozcities(ax, plot_kwargs={"ms": 5, "mfc": "tab:red"})

    fp = FIMG / "ozcities_options.png"
    plt.savefig(fp)


def test_ozcities_proj():
    """ Test plotting oz cities using projection """

    if not HAS_PYPROJ:
        pytest.skip("Missing pyproj module")

    proj = pyproj.Proj("+init=EPSG:3112")

    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  \
                color="k", lw=0.5, fixed_lim=False, proj=proj)
    elems = ozcities(ax, proj=proj)

    fp = FIMG / "ozcities_proj.png"
    plt.savefig(fp)





