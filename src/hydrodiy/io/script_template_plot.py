#!/usr/bin/env python
# -*- coding: utf-8 -*-

[COMMENT]

import sys
import os
import re
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from collections import namedtuple

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from string import ascii_lowercase as letters

import matplotlib as mpl

import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

# Package to plot spatial data
import pyproj
from hydrodiy.gis.oz import ozlayer

from hydrodiy.io import csv, iutils, hyruns

#import importlib.util
#spec = importlib.util.spec_from_file_location("foo", "/path/to/foo.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = [FROOT]
    fdata = [FDATA]
    fimg = [FIMG]
    flogs = froot / "logs" / source_file.stem

    if config.debug:
        fimg = flogs / source_file.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fimg", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fimg, flogs)

    if config.create_folders:
        flogs.mkdir(exist_ok=True, parents=True)
        fimg.mkdir(exist_ok=True, parents=True)

        cext = config.clean_folders_extension
        if cext != "":
            for f in fimg.glob("*." + cext):
                f.unlink()

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(config._asdict())
    logger.info("", nret=1)

    return logger


def get_data(config, script_paths, logger):
    fs = script_paths.fdata / "stations.csv"
    stations, _ = csv.read_csv(fs, index_col="stationid",
                                  dtype={"stationid": str})
    nstations = len(stations)
    logger.info(f"Found {nstations} stations.")

    data = namedtuple("Data", ["stations"])(stations)

    return data


def process(config, script_paths, logger, data):
    nstations = len(data.stations)
    fimg = script_paths.fimg

    logger.info(f"Start plotting", nret=1)

    # To use multipage pdf
    #fpdf = fimg / "images.pdf"
    #pdf = PdfPages(fpdf)

    plt.close("all")

    # Create figure
    fncols, fnrows = config.fncols, config.fnrows
    awidth, aheight = config.awidth, config.aheight
    figsize = (awidth*fncols, aheight*fnrows)
    fig = plt.figure(constrained_layout=True,
                     figsize=figsize)

    # Create mosaic with named axes
    mosaic = [[f"F{fncols*i+j}" for j in range(fncols)] for i in range(fnrows)]
    gw = dict(height_ratios=[1]*fnrows, width_ratios=[1]*fncols)
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)

    nval = 10
    logger.info("Drawing plot")

    for name, ax in axs.items():
        # Retrieve column and row numbers
        iplot = int(name[1:])
        icol = iplot % fncols
        irow = iplot // fncols

        # Get data
        x = pd.date_range("2001-01-01", freq="MS", periods=nval)
        x = x.to_pydatetime()
        y = np.random.uniform(size=nval)

        # Scatter plot
        ax.plot(x, y, "o",
                markersize=10,
                mec="black",
                mfc="pink",
                alpha=0.5,
                label="points")

        # Spatial
        #ozlayer(ax, "ozcoast50m")

        # Decoration
        ax.legend(shadow=True, framealpha=0.7)

        # Axis
        title = f"{name}: Row {irow} / Column {icol}"
        ax.set_title(title)
        ax.set_xlabel("X label")
        ax.set_xlabel("Y label")

    fig.suptitle("Overall title")

    # Footer
    label = f"Generated: {datetime.now().strftime('%H:%M %d/%m/%Y')}"
    fig.text(0.05, 0.010, label, color="#595959", ha="left", fontsize=9)

    # Save file
    fp = fimg / f"image.{config.imgext}"
    fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
    putils.blackwhite(fp)

    # To save to pdf
    #pdf.savefig(fig)
    #pdf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-o", "--overwrite", help="Overwrite data",
                        action="store_true", default=False)
    parser.add_argument("-s", "--sitepattern", help="Site selection pattern",
                        type=str, default=".*")
    args = parser.parse_args()

    # Config
    version = args.version
    debug = args.debug
    imgext = "png"
    fnrows = 2
    fncols = 2
    fdpi = 120
    awidth = 8
    aheight = 8
    ftransparent = False
    create_folders = True
    clean_folders_extension = imgext

    Config = namedtuple("Config",
                        ["version", "debug", "imgext",
                         "fnrows", "fncols", "fdpi",
                         "awidth", "aheight", "ftransparent",
                         "create_folders",
                         "clean_folders_extension"])
    config = Config(version, debug, imgext,
                    fnrows, fncols, fdpi,
                    awidth, aheight, ftransparent,
                    create_folders,
                    clean_folders_extension)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
