#!/usr/bin/env python

## -- Script Meta Data --
## Author  : magpie
## Created : 2017-05-18 20:50:00.002644
## Comment : Generate series of Anderson-Darling and Cramer-Von mises test
##              results
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

nsamples = 50
nval = 30

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# utils functions
def run_test(x, test):
    if test in ['CVM', 'AD']:
        r = robjects.r

        # Run test
        fun = r('function(x){'+\
            'library(goftest);'+\
            test.lower()+'.test(x, "punif", 0, 1)}')
        res = fun(x)

        stat = res[0][0]
        pv = res[1][0]
    else:
        raise ValueError('Test {0} not recognised'.format(test))

    return stat, pv

cc  = ['x{0:02d}'.format(i) for i in range(nval)]
res = pd.DataFrame(np.zeros((nsamples, nval+4)), \
                    columns=cc+['CVM_stat', 'CVM_pvalue', 'AD_stat', 'AD_pvalue'])
for isample in range(nsamples):
    # Sample data
    unif = np.random.uniform(0, 1, nval)
    res.loc[isample, cc] = unif

    # Run test
    for test in ['AD', 'CVM']:
        st, pv = run_test(unif, test)
        res.loc[isample, test+'_stat'] = st
        res.loc[isample, test+'_pvalue'] = pv

fr = os.path.join(froot, 'testdata_AD_CVM.csv')
comments = {'comment': 'Test data for Anderson-Darling (AD) and '+\
        'Cramer-Von Mises (CVM) test of uniformity generated from '+\
        'the goftest R package'}
csv.write_csv(res, fr, comments, source_file, compress=False)


