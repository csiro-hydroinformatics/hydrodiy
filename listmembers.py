#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
import argparse
import pandas as pd
from pathlib import Path
import importlib
from inspect import getmembers, isfunction, isclass, getsourcefile, \
                        isbuiltin, getdoc
from hydrodiy.io import iutils, csv

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="List functions available in hydrodiy package", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-s", "--skipbasemap", \
                    help="Skip code importing basemap package", \
                    action="store_true", default=False)
args = parser.parse_args()
skipbasemap = args.skipbasemap

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent
fhydrodiy = froot / "hydrodiy"

LOGGER = iutils.get_logger(source_file.stem)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
lf = fhydrodiy.glob("**/*.py")

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

members = pd.DataFrame(columns=["package", "module", \
                "object_name", "type", "doc"])

for f in lf:
    if re.search("version|template|test|init", str(f)):
        continue

    LOGGER.info(f"Inspecting {f.name}")

    # Skip if it imports basemap
    if skipbasemap:
        with f.open("r") as fo:
            txt = fo.read()

        if re.search("basemap", txt):
            LOGGER.info("file {f.name} imports basemap, skip.")
            continue

    # Import the file
    modname = f.parts[-2]
    import_name = f"hydrodiy.{modname}.{f.stem}"
    module = importlib.import_module(import_name, package="hydrodiy")

    for obj in getmembers(module):

        # Check object is a class or a function
        if isfunction(obj[1]) or isclass(obj[1]):

            # Check object resides in hydrodiy
            skip = False
            try:
                if not re.search("hydrodiy", getsourcefile(obj[1])):
                    skip = True
            except TypeError:
                skip = True

            if skip:
                continue

            # Get doc
            try:
                doc = re.sub("\\n.*", "", getdoc(obj[1]))
            except TypeError:
                doc = ""

            # Store data
            tp = "class" if isclass(obj[1]) else "function"
            dd = {\
                    "package": "hydrodiy",\
                    "module": import_name,\
                    "object_name": obj[0], \
                    "type": tp, \
                    "doc": doc
            }
            members = members.append(dd, ignore_index=True)


comment = "List of classes and functions in hydrodiy"
fc = os.path.join(froot, "listmembers.html")
pd.set_option("display.max_colwidth", 200)
members.to_html(fc)
#csv.write_csv(members, fc, comment, source_file, compress=False)


