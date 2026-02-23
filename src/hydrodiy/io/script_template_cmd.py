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
    flogs = froot / "logs" / source_file.stem

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs)
    if config.create_folders:
        flogs.mkdir(exist_ok=True, parents=True)

        fout.mkdir(exist_ok=True, parents=True)

        # Clean output folder if needed
        cext = config.clean_folders_extension
        if cext != "":
            for f in fout.glob("*." + cext):
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
    logger.info(f"Found {nstations} stations.")

    data = namedtuple("Data", ["stations"])(stations)

    return data


def process(config, script_paths, logger, data):
    nstations = len(data.stations)
    logger.info(f"Start processing", nret=1)

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
    nbatch = 4
    ibatch = -1 if debug else taskid
    create_folders = True
    clean_folders_extension = ""

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "create_folders",
                         "clean_folders_extension",
                         "sitepattern", "nbatch", "ibatch"])
    config = Config(version, taskid, overwrite,
                    debug, create_folders,
                    clean_folders_extension,
                    sitepattern, nbatch, ibatch)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
