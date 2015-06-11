import sys, os, re, time, string

import gzip

has_distutils = False
try:
    from distutils.sysconfig import get_python_inc
    from distutils.sysconfig import get_python_lib
    has_distutils = True

except ImportError:
    pass

import pandas as pd
import numpy as np

# Max length of comment keys
key_length_max = 30


def _header2comment(header):

    comment = {}
    i = 1

    for s in header:

        if not bool(re.search('-{10}', s)):

            k = re.sub('\:.*$', '', s)
            val = re.sub('%s:' % k, '', s).strip()
            k = re.sub(' +', '_', k.strip().lower())

            if not bool(re.search('\:', s[:key_length_max])):
                k = 'comment_%2.2d' % i
                val = s
                i += 1

            if val != '':
                comment[k] = val

    return comment


def _csvhead(nrow,ncol,comment,src='uknown'):
    """ Produces a nice header for csv files """

    comment_list = comment
    
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append("# --------------------------------------------------")
    
    h.append("# nrow : %d" % nrow)
    h.append("# ncol : %d" % ncol)

    for i in range(len(comment_list)): 
        h.append("# comment%2.2d : %s" %(i, comment_list[i]))

    h.append("# time_written : %s" % time.strftime("%Y-%m-%d %H:%M"))
    
    try:
        h.append("# author : %s" % os.getlogin())
    except:
        h.append("# author : unknwon")

    h.append("# script : %s" % src)
    h.append("# work_dir : %s" % os.getcwd())
    h.append("# python_environment : %s" % os.name)
    h.append("# python_version : %s" % sys.version.replace("\n"," "))
    h.append("# pandas_version : %s" % pd.__version__)
    h.append("# numpy_version : %s" % np.__version__)

    if has_distutils:
        h.append("# python_inc : %s" % get_python_inc())
        h.append("# python_lib : %s" % get_python_lib())
    
    h.append("# --------------------------------------------------")

    return h


def write_csv(data, filename, comment, source,
        index=False,
        compress=True, **kwargs):
    """ write a pandas dataframe to csv with comments """
    
    head = _csvhead(data.shape[0], data.shape[1], 
                comment, src=source)
   
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

    data.to_csv(fcsv, index=index, **kwargs)

    fcsv.close()


def read_csv(filename, has_colnames=True, **kwargs):
    """ Reads data with comments on top to a pandas data frame"""

    # Add gz if file does not exists
    if not os.path.exists(filename):
        filename = '%s.gz' % filename

    # Open proper file type
    if filename.endswith('gz'):
        fcsv = gzip.open(filename, 'rb')
    else:
        fcsv = open(filename, 'r')

    # Reads content
    header = []
    comment = {}

    if has_colnames:

        line = fcsv.readline()

        while line.startswith('#'):
            header.append(re.sub('^# *|\n$', '', line))
            line = fcsv.readline()

        comment = _header2comment(header)
        
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
        data = pd.read_csv(fcsv, names=cns, **kwargs)

        data.columns = [re.sub('\\.','_',cn) 
                        for cn in data.columns] 

    else:
        data = pd.read_csv(fcsv, header=None, **kwargs)

    fcsv.close()

    return data, comment

