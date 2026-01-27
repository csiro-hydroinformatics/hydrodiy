#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-01-27 11:13:54.004090
## Comment : This is a test script
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import namedtuple

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

#import importlib.util
#spec = importlib.util.spec_from_file_location("foo", "/path/to/foo.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)


def get_script_paths(debug, create):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent
    fdata = froot / "data"
    fout = froot / "outputs"
    flogs = froot / "logs"

    if debug:
        fout = flogs / fout.stem

    SP = namedtuple("ScriptPaths",
                    ["source_file", "froot", "fdata", "fout", "flogs"])
    script_paths = SP(source_file, froot, fdata, fout, flogs)

    if create:
        for pa in script_paths:
            if pa.is_file():
                continue
            pa.mkdir(exist_ok=True)

    return script_paths


def get_logger(script_paths):
    basename = script_paths.source_file.stem
    fl = script_paths.flogs / f"{basename}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(vars(args), "Command line arguments")
    logger.started()
    return logger


def get_data(script_paths, logger, nbatch, ibatch, sitepattern):
    fs = script_paths.fdata / "stations.csv"
    allstations, _ = csv.read_csv(fs, index_col="stationid",
                                  dtype={"stationid": str})

    # Select stations
    stations = allstations
    if ibatch < 0:
        idx = allstations.index.str.contains(sitepattern)
        stations = allstations.loc[idx, :]
    else:
        idx = hyruns.get_batch(allstations.shape[0], nbatch, ibatch)
        stations = allstations.iloc[idx, :]

    nstations = len(stations)
    logger.info(f"Dealing with {nstations} stations.")

    data = namedtuple("Data", ["stations"])(stations)

    return data


def process(debug, script_paths, logger, data):
    nstations = len(data.stations)
    logger.info(f"Start processing - Debug={debug}", nret=1)

    for isite, (stationid, sinfo) in enumerate(data.stations.iterrows()):
        ctxt = f"{stationid} ({isite+1}/{nstations})"
        logger.info(f"Processing {ctxt}", nret=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a test script",
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
    taskid = args.taskid
    overwrite = args.overwrite
    debug = args.debug
    sitepattern = args.sitepattern

    # Baseline
    create = True
    script_paths = get_script_paths(debug, create)
    logger = get_logger(script_paths)

    # Data
    nbatch = 4
    ibatch = -1 if debug else taskid
    data = get_data(script_paths, logger, nbatch, ibatch, sitepattern)

    # Process
    process(debug, script_paths, logger, data)

    logger.completed()
