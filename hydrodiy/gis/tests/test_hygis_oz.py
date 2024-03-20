import os
from pathlib import Path
import pytest

from hydrodiy.gis.oz import REGIONS, HAS_PYSHP, ozlayer, ozcities, CAPITAL_CITIES
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


def test_ozcities_kw():
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


def test_ozcities_manual():
    """ Test plotting oz cities using filter """
    plt.close("all")
    fig, ax = plt.subplots()
    lines = ozlayer(ax, "ozcoast50m",  \
                color="k", lw=0.5, fixed_lim=False)

    cities = dict(
        Sydney=[-33.86785, 151.20732], \
        Melbourne=[	-37.814, 144.96332], \
        Brisbane=[-27.46794, 153.02809], \
        Perth=[-31.95224, 115.8614], \
        Adelaide=[-34.92866, 138.59863], \
        Canberra=[-35.28346, 149.12807], \
        Newcastle=[-32.92953, 151.7801], \
        Wollongong=[-34.424, 150.89345], \
        Geelong=[-38.14711, 144.36069], \
        Hobart=[-42.87936, 147.32941], \
        Townsville=[-19.26639, 146.80569], \
        Cairns=[-16.92366, 145.76613], \
        Ballarat=[-37.56622, 143.84957], \
        Toowoomba=[-27.56056, 151.95386], \
        Darwin=[-12.46113, 130.84185], \
        Mandurah=[-32.5269, 115.7217], \
        Mackay=[-21.15345, 149.16554], \
        Bundaberg=[-24.86621, 152.3479]
    )
    c = {n: (cc[1], cc[0]) for n, cc in cities.items()}
    elems = ozcities(ax, cities=c)

    fp = FIMG / "ozcities_manual.png"
    plt.savefig(fp)




