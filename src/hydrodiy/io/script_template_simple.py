#!/usr/bin/env python
# -*- coding: utf-8 -*-

[COMMENT]

import sys, os, re, json, math
import argparse
from pathlib import Path

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

#import importlib.util
#spec = importlib.util.spec_from_file_location("foo", "/path/to/foo.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="[DESCRIPTION]",
                                 formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--version",
                    help="Version number",
                    type=int, required=True)
parser.add_argument("-s", "--sitepattern", help="Site selection pattern",
                    type=str, default="")
parser.add_argument("-i", "--ibatch", help="Batch process number",
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batch processes",
                    type=int, default=7)
parser.add_argument("-t", "--taskid", help="JobID",
                    type=int, default=-1)
parser.add_argument("-p", "--progress", help="Show progress",
                    action="store_true", default=False)
parser.add_argument("-d", "--debug", help="Debug mode",
                    action="store_true", default=False)
parser.add_argument("-o", "--overwrite", help="Overwrite data",
                    action="store_true", default=False)
args = parser.parse_args()

version = args.version
ibatch = args.ibatch
nbatch = args.nbatch
taskid = args.taskid
progress = args.progress
overwrite = args.overwrite
debug = args.debug
sitepattern = args.sitepattern

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = [FROOT]
fdata = [FDATA]
fout = [FOUT]
fimg = [FIMG]

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
flog = froot / "logs" / f"{basename}.log"
flog.parent.mkdir(exist_ok=True)
LOGGER = iutils.get_logger(basename, console=False, contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")

#----------------------------------------------------------------------
# @Get data
#----------------------------------------------------------------------
fs = fdata / "sites.csv"
allsites, _ = csv.read_csv(fs, index_col="siteid")

# Select sites
sites = allsites
if not sitepattern == "":
    idx = allsites.index.str.findall(sitepattern).astype(bool)
    sites = allsites.loc[idx, :]
else:
    if ibatch>=0:
        idx = iutils.get_ibatch(allsites.shape[0], nbatch, ibatch)
        sites = allsites.iloc[idx, :]

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
nsites = len(sites)
for isite, (siteid, sinfo) in enumerate(sites.iterrows()):

    LOGGER.context = f"{siteid} ({isite+1}/{nsites})"

    LOGGER.info("Processing")

LOGGER.completed()

