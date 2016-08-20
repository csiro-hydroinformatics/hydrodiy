#!/usr/bin/env python

# -- Script Meta Data --
# Author : J. Lerat, EHP, Bureau of Meteorogoloy
# Versions :
#    V00 - Script written from template on 2016-03-29 10:52:53.464102
#
# ------------------------------

import sys, os, re, json, math
import subprocess

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from itertools import product as prod

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
def set_config():
    ''' Set script configuration '''

    # Script args
    #nargs = len(sys.argv)
    #if nargs > 0:
    #    arg1 = sys.argv[1]

    # Script path
    source_file = os.path.abspath(__file__)
    froot = os.path.dirname(source_file)

    fdata = os.path.join(froot, 'data')
    #if not os.path.exists(fdata):
    #    os.mkdir(fdata)
    fout = froot

    config = {
        'ibatch': 0,
        'nbatch': 5,
        'source_file': source_file,
        'start': '{0}'.format(datetime.now()),
        'froot': froot,
        'fout': fout,
        'fdata': fdata
    }

    # Set instance of logger
    select = ['ibatch', 'nbatch']
    vartxt = iutils.vardict2str({key:config[key] for key in select})
    flog = os.path.join(fout, re.sub('\\.py.*', '', source_file)+ \
                                        '_'+vartxt+'.log')

    log_name = re.sub('\\..*', '', os.path.basename(source_file))

    LOGGER = iutils.get_logger(log_name, level='INFO', \
        fmt='%(asctime)s - %(message)s', \
        console=True, flog=flog)

    return config, LOGGER


#----------------------------------------------------------------------
def process(config, LOGGER):
    ''' Process script '''

    # I -- GET DATA ---------------
    #fd = os.path.join(FDATA, 'data.csv')
    #data, comment = csv.read_csv(fd)

    # Extract site batch
    sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

    # Extract batch of sites
    ibatch = config['ibatch']
    nbatch = config['nbatch']
    idx = iutils.get_ibatch(len(sites), nbatch, ibatch)
    sites = sites.iloc[idx]

    mess = '{0} sites found'.format(sites.shape[0])
    LOGGER.info(mess)

    # II -- PROCESS ---------------
    nsites = sites.shape[0]
    count = 0

    # To run a bash command
    cmd = 'ls -al'
    subprocess.check_call(cmd, shell=True)

    # To run process across sites
    for idx, row in sites.iterrows():

        siteid = idx
        count += 1

        mess = 'Dealing with site {0} ({1:3d}/{2:3d})'.format(siteid, \
                    count, nsites)
        LOGGER.info(mess)

    # III -- STORE DATA ---------------


#----------------------------------------------------------------------
def entry_point():
    ''' Main function of the script. No need to edit that section. '''

    # Define config options
    config, LOGGER = set_config()

    # Process data
    process(config, LOGGER)
    LOGGER.info('Process completed')

    LOGGER.info('Script {0} completed'.format( \
        config['source_file']))


#----------------------------------------------------------------------
if __name__ == "__main__":
    entry_point()

