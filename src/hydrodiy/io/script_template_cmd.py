#!/usr/bin/env python
# -*- coding: utf-8 -*-

[COMMENT]

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


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = [FROOT]
    fdata = [FDATA]
    fout = [FOUT]
    flogs = froot / "logs"

    if debug:
        fout = flogs / fout.stem

    SP = namedtuple("ScriptPaths",
                    ["source_file", "froot", "fdata", "fout", "flogs"])
    script_paths = SP(source_file, froot, fdata, fout, flogs)

    for pa in script_paths:
        if pa.is_file():
            continue
        pa.mkdir(exist_ok=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.source_file.stem
    fl = script_paths.flogs / f"{basename}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    logger.log_dict(vars(args), "Command line arguments")
    logger.started()
    return logger


def get_data(config, script_paths, logger):
    fs = script_paths.fdata / "stations.csv"
    allstations, _ = csv.read_csv(fs, index_col="stationid",
                                  dtype={"stationid": str})

    # Select stations
    stations = allstations
    if ibatch < 0:
        idx = allstations.index.str.contains(config.sitepattern)
        stations = allstations.loc[idx, :]
    else:
        idx = hyruns.get_batch(allstations.shape[0],
                               config.nbatch,
                               config.ibatch)
        stations = allstations.iloc[idx, :]

    nstations = len(stations)
    logger.info(f"Dealing with {nstations} stations.")

    data = namedtuple("Data", ["stations"])(stations)

    return data


def process(config, script_paths, logger, data):
    nstations = len(data.stations)
    logger.info(f"Start processing - Debug={config.debug}", nret=1)

    for isite, (stationid, sinfo) in enumerate(data.stations.iterrows()):
        ctxt = f"{stationid} ({isite+1}/{nstations})"
        logger.info(f"Processing {ctxt}", nret=1)


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
    taskid = args.taskid
    overwrite = args.overwrite
    debug = args.debug
    sitepattern = args.sitepattern

    clear = False
    nbatch = 1
    ibatch = 0

    CF = namedtuple("Config",
                    ["version", "taskid",
                     "overwrite", "debug",
                     "sitepattern", "clear",
                     "nbatch", "ibatch"])
    config = CF(version, taskid, overwrite, debug,
                sitepattern, clear, nbatch, ibatch)

    # Get data
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
