#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-01-27 11:13:52.480538
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

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

#import importlib.util
#spec = importlib.util.spec_from_file_location("foo", "/path/to/foo.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
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
parser.add_argument("-s", "--stationpattern", help="Site selection pattern",
                    type=str, default=".*")
args = parser.parse_args()

version = args.version
taskid = args.taskid
overwrite = args.overwrite
debug = args.debug
stationpattern = args.stationpattern

nbatch = 4
ibatch = -1 if debug else 0

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent
fdata = froot / "data"
fdata.mkdir(exist_ok=True)

fout = froot / "outputs"
fout.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / f"{basename}.log"
flog.parent.mkdir(exist_ok=True)
LOGGER = iutils.get_logger(basename, console=True,
                           flog=flog)
LOGGER.log_dict(vars(args), "Command line arguments")

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
fs = fdata / "stations.csv"
allstations, _ = csv.read_csv(fs, index_col="stationid",
                              dtype={"stationid": str})

# Select stations
stations = allstations
if not stationpattern == "":
    idx = allstations.index.str.contains(stationpattern)
    stations = allstations.loc[idx, :]
else:
    idx = hyruns.get_batch(allstations.shape[0], nbatch, ibatch)
    stations = allstations.iloc[idx, :]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
nstations = len(stations)
for istation, (stationid, sinfo) in enumerate(stations.iterrows()):

    ctxt = f"{stationid} ({istation+1}/{nstations})"
    LOGGER.info(f"Processing {ctxt}", nret=1)


LOGGER.completed()

