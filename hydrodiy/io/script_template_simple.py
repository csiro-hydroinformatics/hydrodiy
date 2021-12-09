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

from tqdm import tqdm

#import importlib.util
#spec = importlib.util.spec_from_file_location("foo", "/path/to/foo.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="A script", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--version", \
                    help="Version number", \
                    type=int, required=True)
parser.add_argument("-p", "--sitepattern", help="Site selection pattern", \
                    type=str, default="")
parser.add_argument("-i", "--ibatch", help="Batch process number", \
                    type=int, default=-1)
parser.add_argument("-n", "--nbatch", help="Number of batch processes", \
                    type=int, default=7)
parser.add_argument("-j", "--jobid", help="JobID", \
                    type=int, default=-1)
parser.add_argument("-p", "--progress", help=" Show progress", \
                    action="store_true", default=False)
args = parser.parse_args()

version = args.version
ibatch = args.ibatch
nbatch = args.nbatch
jobid = args.jobid
progress = args.jobid
sitepattern = args.sitepattern

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = [FROOT]
fdata = [FDATA]
fout = [FOUT]
fimg = [FIMG]

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

for arg in vars(args):
    LOGGER.info("{0} script argument {1} = {2}".format(basename, \
                            arg, getattr(args, arg)))

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
fs = fdata / "sites.csv"
sites_all, _ = csv.read_csv(fs, index_col="siteid")

# Select sites
sites = sites_all
if not sitepattern == "":
    idx = sites_all.index.str.findall(sitepattern).astype(bool)
    sites = sites_all.loc[idx, :]
else:
    if args.ibatch >=0:
        idx = iutils.get_ibatch(sites_all.shape[0], nbatch, ibatch)
        sites = sites_all.iloc[idx, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for i, (siteid, row) in enumerate(sites.iterrows()):

    LOGGER.info("dealing with {0} ({1}/{2})".format( \
        siteid, i, len(sites)))



LOGGER.info("Process completed")

