#!/usr/bin/env python

import sys, os, re, json, math
import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

# Define config options
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------




LOGGER.info('Process completed')

