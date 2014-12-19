import sys
import os
import re
import time
import string
import gzip
import pandas as pd

def _csvhead(nrow,ncol,comment,src='uknown'):
    """ Produces a nice header for csv files """
    comment_list = comment
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append("# --------------------------------------------------")
    h.append("# File comments :")
    h.append("# nrow = %d ncol= %d"%(nrow,ncol))
    for co in comment_list: 
        h.append("# %s"%co)
    h.append("# ")
    h.append("# File info :")
    h.append("# Written on : %s"%time.strftime("%Y-%m-%d %H:%M"))
    try:
        h.append("# Author : %s"%os.getlogin())
    except (OSError, AttributeError):
        h.append("# Author : ")
    h.append("# Produced with python script : %s"%src)
    h.append("# Python work dir : %s"%os.getcwd())
    h.append("# Python environment : %s"%os.name)
    h.append("# Python version : %s"%sys.version.replace("\n"," "))

    return h

def write_csv(data, filename, comment, index=False, source='unknown', compress=True):
    """ write a pandas dataframe to csv with comments """
    
    head = _csvhead(data.shape[0], data.shape[1], comment, src=source)
   
    # defines file name
    filename_full = filename
    if compress and ~filename.endswith('.gz'):
        filename_full += '.gz'
    
    if compress:
        fcsv = gzip.open(filename_full, 'wb')
    else:
        fcsv = open(filename_full, 'w')

    # Write data to file
    for line in head:
        fcsv.write('%s\n'%line)

    data.to_csv(fcsv, index=index)

    fcsv.close()


def read_csv(filename, has_header=True):
    """ Reads data with comments on top to a pandas data frame"""

    # Open proper file type
    if filename.endswith('gz'):
        fcsv = gzip.open(filename, 'rb')
    else:
        fcsv = open(filename, 'r')

    # Reads content
    comment = []
    if has_header:
        line = fcsv.readline()
        while line.startswith('#'):
            comment.append(re.sub('^# *|\n$', '', line))
            line = fcsv.readline()
        
        # deals with multi-index columns
        # reformat columns like (idx1-idx2-...)
        se = re.findall('\"\([^\(]*\)\"', line)
        linecols = line
        if se:
            for cn in se:
                cn2 = re.sub('\(|\)|\"|\'','',cn)
                cn2 = re.sub(', *','-',cn2)
                linecols = re.sub(re.escape(cn), cn2, linecols)
        cns = string.strip(linecols).split(',') 

        # Reads data with proper column names    
        data = pd.read_csv(fcsv, names=cns)
        data.columns = [re.sub('\\.','_',cn) 
                        for cn in data.columns] 
    else:
        data = pd.read_csv(fcsv, header=None)

    fcsv.close()

    return data, comment

