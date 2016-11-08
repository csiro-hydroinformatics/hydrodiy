#!/usr/bin/env python


import sys, os, re, json, math

from datetime import datetime

from dateutil.relativedelta import relativedelta as delta
from string import ascii_lowercase as letters
from calendar import month_abbr as months

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib.backends.backend_pdf import PdfPages

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from hydrodiy.gis.oz import Oz


#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Plot dimensions
fnrows = 2
fncols = 2
fdpi = 100
awidth = 1000
aheight = 1000

# Set matplotlib options
putils.set_mpl()

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.join(os.path.dirname(source_file), '..')

fdata = os.path.join(froot, 'data')
fimg = os.path.join(froot, 'images')
if not os.path.exists(fimg): os.mkdir(fimg)

# Set instance of logger
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

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

# To use multipage pdf
#fpdf = os.path.join(FOUT, 'evap_sensitivity.pdf')
#pdf = PdfPages(fpdf)

plt.close('all')

fig = plt.figure()

gs = gridspec.GridSpec(fnrows, fncols,
        width_ratios=[1] * fncols,
        height_ratios=[1] * fnrows)

nval = 100

for i in range(fnrows*fncols):

    icol = i%fncols
    irow = i/fncols
    ax = fig.add_subplot(gs[irow, icol])

    # To use oz
    #om = Oz(ax=ax)

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
    ax.legend(shadow=True,
        framealpha=0.7)

    ax.set_title('Title')
    ax.set_xlabel('X label')
    ax.set_xlabel('Y label')

fig.suptitle('Overall title')

# Footer
label = 'Generated: %s' % datetime.now().strftime('%H:%M %d/%m/%Y')
fig.text(0.05, 0.010, label, color='#595959', ha='left', fontsize=9)

# Save figure
fig.set_size_inches(float(fncols * awidth)/fdpi,
                float(fnrows * aheight)/fdpi)

gs.tight_layout(fig)

fp = os.path.join(fimg, 'image.png')
fig.savefig(fp, dpi=fdpi)

# To save to pdf
#pdf.savefig(fig)
#pdf.close()

