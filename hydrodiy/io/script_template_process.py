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
def get_config():
    ''' Get script configuration '''

    config = {}

    #nargs = len(sys.argv)
    #if nargs > 0:
    #    arg1 = sys.argv[1]

    config['ibatch'] = 0
    config['nbatch'] = 5

    return config


#----------------------------------------------------------------------
def set_folders(config):
    ''' Set script folders '''

    source_file = os.path.abspath(__file__)

    froot = os.path.dirname(source_file)

    fdata = os.path.join(froot, 'data')
    if not os.path.exists(fdata):
        os.mkdir(fdata)

    config['source_file'] = froot
    config['froot'] = froot
    config['fdata'] = fdata


#----------------------------------------------------------------------
def get_logger(config):
    ''' Set instance of logger '''
    source_file = config['source_file']
    flog = source_file + '.log'
    log_name = re.sub('\\..*', '', os.path.basename(source_file))
    LOGGER = iutils.get_logger(log_name, level='INFO', \
        fmt='%(asctime)s - %(message)s', \
        console=True, flog=flog)

    return LOGGER


#----------------------------------------------------------------------
def get_data(config, LOGGER):
    ''' Retrive data '''

    data = {}
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
    data = {'sites': sites}

    return data


#----------------------------------------------------------------------
def process(config, LOGGER, data):
    ''' Process data '''

    sites = data['sites']
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


#----------------------------------------------------------------------
def store(config, LOGGER, data):
    ''' Write results to disk '''

    pass

#----------------------------------------------------------------------
def entrypoint():
    ''' Main function of the script. No need to edit that section. '''

    # Define config options
    config = get_config()

    # Set path to folders
    set_folders(config)
    source_file = config['source_file']

    # Get logger object
    LOGGER = get_logger(config)
    LOGGER.info('Script {0} started'.format(source_file))

    # Get data
    data = get_data(config, LOGGER)
    LOGGER.info('Data extracted')

    # Process data
    process(config, LOGGER, data)
    LOGGER.info('Process completed')

    # Store data
    store(config, LOGGER, data)
    LOGGER.info('Storage completed')

    LOGGER.info('Script {0} completed'.format(source_file))


#----------------------------------------------------------------------
if __name__ == "__main__":
    entrypoint()

