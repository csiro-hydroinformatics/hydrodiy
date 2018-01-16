#!/usr/bin/env python

## -- Script Meta Data --
## Author  : magpie
## Created : 2017-05-18 20:50:00.002644
## Comment : Generate table of Cramer Von Mises pvalues
##
## ------------------------------
import os, sys
import numpy as np
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

npvalues = 1000
nsamples = range(21) + range(25, 55, 5) + range(60, 510, 10) + \
                range(550, 1050, 50) + range(1100, 10100, 100) + \
                []

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# utils functions
def cvm_test(x):
    r = robjects.r

    # Run test
    fun = r('function(x){'+\
        'library(goftest);'+\
        test.lower()+'.test(x, "punif", 0, 1)}')
    res = fun(x)

    stat = res[0][0]
    pv = res[1][0]

    return stat, pv

fr = os.path.join(froot, 'cramer_von_mises_test_pvalues.csv')
comments = {'comment': 'Test data for Anderson-Darling (AD) and '+\
        'Cramer-Von Mises (CVM) test of uniformity generated from '+\
        'the goftest R package'}
csv.write_csv(res, fr, comments, source_file, compress=False, \
                float_format='%0.12f')


