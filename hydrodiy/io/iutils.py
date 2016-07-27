import sys
import os
import re
import gzip
from datetime import datetime
import logging

import numpy as np

def random_password(length=10, chr_start=35, chr_end=128):
    ''' Generate random password

    Parameters
    -----------
    length : int
        Number of characters
    chr_start : int
        Ascii code defining the start of allowed characters
    chr_end : int
        Ascii code defining the end of allowed characters

    Returns
    -----------
    pwd : str
        Password

    Example
    -----------
    >>> pwd = iutils.random_password()
    '''

    pwd = ''.join([chr(i)
        for i in np.random.randint(chr_start, chr_end, size=length)])

    return pwd



def find_files(folder, pattern, recursive=True):
    ''' Find files recursively based on regexp pattern search

    Parameters
    -----------
    folder : str
        Folder to be searched
    pattern : str
        Regexp pattern to be used. See re.search function
    recursive : bool
        Search folder recursively or not

    Returns
    -----------
    found : list
        List of filenames

    Example
    -----------
    Look for all python scripts in current folder
    >>> lf = iutils.find_files('.', '.*\\.py', False)
    '''

    found = []

    if recursive:
        for root, dirs, files in os.walk(folder):
            for filename in files:
                fn = os.path.join(root, filename)
                if not re.search(pattern, fn) is None:
                    found.append(fn)
    else:
        files = next(os.walk(folder))[2]
        for filename in files:
            fn = os.path.join(folder, filename)
            if not re.search(pattern, fn) is None:
                found.append(fn)

    return found


def vardict2str(data):
    ''' Convert a dict to a string with the format v1[value1]_v2[value2]

    Parameters
    -----------
    data : dict
        Non nested dictionary containing data

    Returns
    -----------
    vars : str
        String containing variables

    Example
    -----------
    >>> iutils.vardict2string({'name':'bob', 'phone':2010})

    '''
    out = []
    for k, v in data.iteritems():
        out.append('{0}[{1}]'.format(k, v))

    return '_'.join(out)


def str2vardict(source):
    ''' Find match in the form v1[value1]_v2[value2] in the
    source string and returns a dict with the value found

    Parameters
    -----------
    source : str
        String to search in
    varnames : list
        List of variable names to be searched

    Example
    -----------
    >>> source = 'name[bob]_phone[2010]'
    >>> iutils.str2vardict(source)

    '''

    out = {}

    se = re.findall('[^_]+\\[[^\\[]+\\]', source)

    for vn in se:
        # Get name
        name = re.sub('\[.*', '', vn)

        # Get value and attempt conversion
        value = re.sub('.*\[|\]', '', vn)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

        out[name] = value

    return out


def script_template(filename,
        type='process',
        author='J. Lerat, EHP, Bureau of Meteorogoloy'):
    ''' Write a script template

    Parameters
    -----------
    filename : str
        Filename to write the script to
    type : str
        Type of script:
        'process' is a data processing script
        'plot' is plotting script
    author : str
        Script author

    Example
    -----------
    >>> iutils.script_template('a_cool_script.py', 'plot', 'Bob Marley')

    '''
    FMOD, modfile = os.path.split(__file__)
    f = os.path.join(FMOD, 'script_template_%s.py' % type)
    with open(f, 'r') as ft:
        txt = ft.readlines()

    meta = ['# -- Script Meta Data --\n']
    meta += ['# Author : %s\n' % author]
    meta += ['# Versions :\n']
    meta += [('#    V00 - Script written from template '
                    'on %s\n') % datetime.now()]
    meta += ['#\n', '# ------------------------------\n']

    txt = txt[:2] + meta + txt[3:]

    with open(filename, 'w') as fs:
        fs.writelines(txt)


def get_logger(name, level, console=False, flog=None,
        check_duplicate=True):
    ''' Get a logger

    Parameters
    -----------
    name : str
        Logger name
    level : str
        Logging level.
    console : bool
        Log to console
    flog : str
        Path to log file. If none, no log file is used.
    check_duplicate : bool
        Check if console or flog are already attached to logger
    '''

    logger = logging.getLogger(name)

    if not level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError('{0} not a valid level'.format(level))
    logger.setLevel(getattr(logging, level))

    # log format
    ft = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check handlers
    has_console = False
    has_flog = False
    for hd in logger.handlers:
        if isinstance(hd, logging.StreamHandler):
            has_console = True

        if isinstance(hd, logging.FileHandler):
            if hd.baseFilename == flog:
                has_flog = True

    # log to console
    if console and not has_console:
        sh = logging.StreamHandler()
        sh.setFormatter(ft)
        logger.addHandler(sh)

    # log to file
    if not flog is None and not has_flog:
        fh = logging.FileHandler(flog)
        fh.setFormatter(ft)
        logger.addHandler(fh)

    return logger


def get_ibatch(nsites, nbatch, ibatch):
    ''' Returns the indices of sites within a batch

    Parameters
    -----------
    nsites : int
        Number of sites
    nbatch : int
        Number of batches
    ibatch : int
        Batch index (from 0 to nbatch-1)

    Returns
    -----------
    idx : list
        List of integer containing sites indexes (0 = first site)

    Example
    -----------
    >>>  idx = iutils.get_ibatch(20, 2, 0)
    '''

    nsites_batch = nsites/nbatch
    if nsites_batch == 0:
        raise ValueError('Number of sites per batch is 0 (nsites={0}, nbatch={1})'.format(
            nsites, nbatch))

    start = nsites_batch * ibatch
    idx = np.arange(start, start+nsites_batch)
    idx = list(idx[idx<nsites])

    return idx
