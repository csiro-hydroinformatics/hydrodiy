#!/usr/bin/env python


import sys, os, re, json, math

from datetime import datetime

from itertools import product as prod

from dateutil.relativedelta import relativedelta as delta
from string import ascii_lowercase as letters
from calendar import month_abbr as months

import numpy as np
import pandas as pd

import matplotlib as mpl

# Desable X11 running
if mpl.get_backend() != 'Agg':
    mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

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
    fimg = os.path.join(froot, 'images')
    for folder in [fdata, fimg]:
        if not os.path.exists(folder):
            pass
            #os.mkdir(folder)

    config = {
        'ibatch': 0,
        'nbatch': 5,
        'source_file': source_file,
        'start': '{0}'.format(datetime.now()),
        'froot': froot,
        'fdata': fdata,
        'fimg': fimg,
        'fig_nrows':2,
        'fig_ncols':2,
        'fig_dpi':100,
        'ax_width':1000,
        'ax_height':1000
    }

    # Set instance of logger
    select = ['ibatch', 'nbatch']
    vartxt = iutils.vardict2str({key:config[key] for key in select})
    basename = re.sub('\\.py.*', '', os.path.basename(source_file))
    flog = os.path.join(fimg, basename + '_'+vartxt+'.log')
    LOGGER = iutils.get_logger(basename, flog=flog)

    return config, LOGGER


#-------------------------------------------------------------------
def process(config, LOGGER):
    ''' Function producing the plots '''

    #------------------------------------------------------------
    # Get data
    #------------------------------------------------------------
    #fd = '%s/data.csv' % FDATA
    #data, comment = csv.read_csv(fd)

    sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

    mess = '{0} sites found'.format(sites.shape[0])
    LOGGER.info(mess)

    #------------------------------------------------------------
    # Plot
    #------------------------------------------------------------
    fig_nrows = config['fig_nrows']
    fig_ncols = config['fig_ncols']

    # To use multipage pdf
    #fpdf = os.path.join(FOUT, 'evap_sensitivity.pdf')
    #pdf = PdfPages(fpdf)
    #pdf.savefig(fig)
    #pdf.close()

    plt.close('all')

    fig = plt.figure()

    gs = gridspec.GridSpec(fig_nrows, fig_ncols,
            width_ratios=[1] * fig_ncols,
            height_ratios=[1] * fig_nrows)

    nval = 100

    for i, j in prod(range(fig_nrows), range(fig_ncols)):

        ax = fig.add_subplot(gs[i, j])

        xx = np.random.uniform(size=(nval, 2))
        x = xx[:,0]
        y = xx[:,1]

        # Scatter plot
        ax.plot(x, y, 'o',
            markersize=10,
            mec='black',
            mfc='pink',
            alpha=0.5,
            label='points')

        # Decoration
        ax.legend(frameon=True,
            shadow=True,
            fancybox=True,
            framealpha=0.7,
            numpoints=1)

        ax.set_title('Title')
        ax.set_xlabel('X label')
        ax.set_xlabel('Y label')

    fig.suptitle('Overall title')

    ax_width = config['ax_width']
    ax_height = config['ax_height']
    fig_dpi = config['fig_dpi']
    fig.set_size_inches(float(fig_ncols * ax_width)/fig_dpi,
                    float(fig_nrows * ax_height)/fig_dpi)

    gs.tight_layout(fig)

    fimg = config['fimg']
    fp = os.path.join(fimg, 'image.png')
    fig.savefig(fp, dpi=fig_dpi)



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

