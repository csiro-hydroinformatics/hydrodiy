#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-08-02 Thu 11:23 AM
## Comment : Generate ensemble verification metrics
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime

from hydrodiy.io import iutils
from hydrodiy.stat import metrics, transform

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Configure obs generation
nvar = 5
nval = 1000

# Configure times series obs generation
nens = 1000
nval = 50

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get obs
#----------------------------------------------------------------------

# Create a random set of forecasts
fcst = np.random.normal(loc=1, size=(nval, nens))
fcst = fcst + 0.5*np.sin(np.linspace(-math.pi, math.pi, nval))[:, None]
fcst = np.maximum(fcst, 0.)

days = pd.date_range('2010-01-01', periods=nval)
fcst = pd.DataFrame(fcst, columns=['E{0}'.format(i) for i in range(nens)])

# Create a random set of obs
obs = fcst.mean(axis=1) + np.random.normal(size=nval)
obs = np.maximum(obs, 0.)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# CRPS score
crps, crps_table = metrics.crps(obs, fcst)
LOGGER.info('CRPS ={0:0.3f}'.format(crps.crps))

# CRPS skill score
crps_ss = (1-crps.crps/crps.uncertainty)*100
LOGGER.info('CRPS_SS ={0:0.3f}'.format(crps_ss))

# Reliability (Pvalue of Cramer-Von Mises test for Probability Integral
# Transform data)
stat, CVpvalue = metrics.alpha(obs, fcst)
LOGGER.info('ALPHA - CV ={0:0.3f}'.format(CVpvalue))

# .. same with Kolmogorov-Smirnov test
stat, KSpvalue = metrics.alpha(obs, fcst, type='KS')
LOGGER.info('ALPHA - KS ={0:0.3f}'.format(KSpvalue))

# Mean ensemble bias
bias = metrics.bias(obs, np.mean(fcst, axis=1))
LOGGER.info('Bias ={0:0.3f}'.format(bias))

# Mean ensemble rank correlation correlation
corr = metrics.corr(obs, np.mean(fcst, axis=1), type='Spearman')
LOGGER.info('Rk Corr ={0:0.3f}'.format(corr))

# Computing metrics in transformed space
# .. using a log transform with a shift parameter set to nu=1e-2
# (see hydrodiy.stat.transform)
trans = transform.Log()
trans.nu = 1e-2

biaslog = metrics.bias(obs, np.mean(fcst, axis=1), trans=trans)
LOGGER.info('BiasLog ={0:0.3f}'.format(biaslog))

corrlog = metrics.corr(obs, np.mean(fcst, axis=1), type='Spearman', trans=trans)
LOGGER.info('CorrLog ={0:0.3f}'.format(corrlog))



LOGGER.info('Process completed')

