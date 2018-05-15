import sys, os, re

import shlex
import subprocess

from datetime import datetime
import logging
import stat

from hydrodiy import PYVERSION

if PYVERSION == 2:
    from StringIO import StringIO
elif PYVERSION == 3:
    from io import StringIO

from io import BytesIO

import requests

import numpy as np

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

    if not os.path.exists(folder):
        raise ValueError('Folder {0} does not exists'.format(folder))

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


def dict2str(data, prefix=None):
    ''' Convert a dict to a string with the format v1[value1]_v2[value2]

    Parameters
    -----------
    data : dict
        Non nested dictionary containing data
    prefix : str
        Prefix to be added at the beginning of the string

    Returns
    -----------
    vars : str
        String containing variables with prefix

    Example
    -----------
    >>> iutils.dict2string({'name':'bob', 'phone':2010})

    '''
    out = []

    # Add items
    for key in sorted(data):
        out.append('{0}[{1}]'.format(key, data[key]))

    out = '_'.join(out)

    # Add prefix if needed
    if not prefix is None:
        if prefix!='':
            out = prefix + '_' + out

    return out


def str2dict(source, num2str=True):
    ''' Find match in the form v1[value1]_v2[value2] in the
    source string and returns a dict with the value found

    Parameters
    -----------
    source : str
        String to search in
    num2str : bool
        Convert all value to string

    Example
    -----------
    >>> source = 'name[bob]_phone[2010]'
    >>> iutils.str2dict(source)

    '''

    # Excludes path and file extension
    source = re.sub('\\.[^\\.]+$', '', os.path.basename(source))
    prefix = source

    # Search for pattern match
    out = {}
    search = re.findall('[^_]+\\[[^\\[]+\\]', source)

    for match in search:
        # Get name
        name = re.sub('\[.*', '', match)

        # Get value
        value = re.sub('.*\[|\]', '', match)

        # Remove item from prefix
        prefix = re.sub('_*'+name+'\\[' + value + '\\]', '', prefix)

        # Attempt conversion if required
        if not num2str:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        out[name] = value

    return out, prefix


def script_template(filename, comment,
        stype='simple',
        author=None):
    ''' Write a script template

    Parameters
    -----------
    filename : str
        Filename to write the script to
    comment : str
        Comment on purpose of the script
    stype : str
        Type of script:
        * simple: script with minimal functionalities
        * plot: plotting script
    author : str
        Script author

    Example
    -----------
    >>> iutils.script_template('a_cool_script.py', 'Testing', 'plot', 'Bob Marley')

    '''
    if not stype in ['simple', 'plot']:
        raise ValueError('Script type {0} not recognised'.format(stype))

    # Open script template
    FMOD = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(FMOD, 'script_template_{0}.py'.format(stype))
    with open(f, 'r') as ft:
        txt = ft.readlines()

    if author is None:
        try:
            author = os.getlogin()
        except:
            author = 'unknown'

    meta = ['## -- Script Meta Data --\n']
    meta += ['## Author  : {0}\n'.format(author)]
    meta += ['## Created : {0}\n'.format(datetime.now())]
    meta += ['## Comment : {0}\n'.format(comment)]
    meta += ['##\n', '## ------------------------------\n']

    txt = txt[:3] + meta + txt[3:]

    if PYVERSION == 2:
        txt = [re.sub('os\.makedirs\(fimg, exist_ok\=True\)', \
            'if not os.path.exists(fimg): os.makedirs(fimg)', line) \
                    for line in txt]

        txt = [re.sub('os\.makedirs\(fout, exist_ok=True\)', \
            'if not os.path.exists(fout): os.makedirs(fout)', line) \
                    for line in txt]

    with open(filename, 'w') as fs:
        fs.writelines(txt)

    # Make it executable for the user
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


def get_logger(name, level='INFO', \
        console=True, flog=None, \
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
        overwrite=True,
        excepthook=True,
        no_duplicate_handler=True):
    ''' Get a logger object

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
    fmt : str
        Log format
    overwrite : bool
        If true, removes the flog files if exists
    excepthook : bool
        Change sys.excepthook to trap all errors in the logger
        (does not work within IPython)
    no_duplicate_handler : bool
        Avoid duplicating console or flog log handlers

    Returns
    -----------
    logger : logging.Logger
        Logger instance
    '''

    logger = logging.getLogger(name)

    # Set logging level
    if not level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError('{0} not a valid level'.format(level))

    logger.setLevel(getattr(logging, level))

    # Set logging format
    ft = logging.Formatter(fmt)

    # Check handlers
    has_console = False
    has_flog = False
    if no_duplicate_handler:
        for hd in logger.handlers:
            if isinstance(hd, logging.StreamHandler):
                has_console = True

            if isinstance(hd, logging.FileHandler):
                if hd.baseFilename == flog:
                    has_flog = True

    # log to console
    if console and not has_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ft)
        logger.addHandler(sh)

    # log to file
    if not flog is None and not has_flog:
        if overwrite:
            if os.path.exists(flog): os.remove(flog)

        fh = logging.FileHandler(flog)
        fh.setFormatter(ft)
        logger.addHandler(fh)

    if excepthook:
        def catcherr(exc_type, exc_value, exc_traceback):
            logger.error('Unexpected error', exc_info=(exc_type, exc_value, exc_traceback))
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = catcherr

    # Close all handlers
    [h.close() for h in logger.handlers]

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
    >>>  idx = iutils.get_ibatch(20, 5, 1)
    [4, 5, 6, 7]
    >>>  idx = iutils.get_ibatch(20, 5, 2)
    [8, 9, 10, 11]

    '''
    if nsites < 1:
        raise ValueError('Number of sites lower than 1')

    if nsites < nbatch:
        raise ValueError('Number of sites lower than number of batches')

    if ibatch < 0 or ibatch >= nbatch:
        raise ValueError('Expected ibatch in [0, {0}], got {1}'.format(\
            nbatch-1, ibatch))

    nsites_per_batch = nsites//nbatch+1
    if nsites_per_batch == 0:
        raise ValueError('Number of sites per batch is 0'+\
            ' (nsites={0}, nbatch={1})'.format(
                nsites, nbatch))

    if nsites_per_batch > nsites:
        raise ValueError(('Number of sites per batch({0})'+\
            ' is greater than nsites({1}))').format(
                nsites_per_batch, nsites))

    start = nsites_per_batch * ibatch
    if start > nsites-1:
        raise ValueError(('Batch index({0}) is too large for '+\
            ' the number of sites({1}) and number of sites '+\
            'per batch ({2})').format(
                ibatch, nsites, nsites_per_batch))

    return np.arange(start, min(nsites, start+nsites_per_batch))


def download(url, filename=None, logger=None, nprint=5, \
        user=None, pwd=None, timeout=None):
    ''' Download file by chunk. Appropriate for large files

    Parameters
    -----------
    url : str
        File URL
    filename : str
        Output file path. If None returns a pipe.
    logger : logging.Logger
        Logger instance
    nprint : int
        Frequency of logger printing in Mb
    user : str
        User name
    pwd : str
        Password

    Returns
    -----------
    stream : io.BytesIO
        Binary stream to download data
    '''

    auth = None
    if not user is None:
        auth = requests.auth.HTTPBasicAuth(user, pwd)

    # Run request
    if timeout is None:
        req = requests.get(url, auth=auth)
    else:
        req = requests.get(url, auth=auth, timeout=timeout)

    # Raise error if HTTP problem
    req.raise_for_status()

    # Function to download data
    def get_data(fobj):
        count = 0
        for chunk in req.iter_content(chunk_size=1024):
            count += 1
            if count % nprint*1000 == 0 and not logger is None:
                logger.info('{0} - chunk {1}'.format(\
                    os.path.basename(filename), count))

            if chunk: # filter out keep-alive new chunks
                fobj.write(chunk)

    # Store data in pipe
    if filename is None:
        stream = BytesIO()
        get_data(stream)

        # Rewind at the start of the file
        stream.seek(0)
        return stream

    else:
        # Store data in file
        with open(filename, 'wb') as fobj:
            get_data(fobj)

        return None


def run_command(cmd, logger, prefix='cmd ~ ', shell=False):
    ''' Run command line and save outputs to a logger

    Parameters
    -----------
    cmd : str
        Command line to execute
    logger : logging.Logger
        Logger to be used
    prefix : str
        Prefix to be added at the beginning of the log
        and err messages
    '''
    # Start subprocess
    args = shlex.split(cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, \
                            stderr=subprocess.PIPE, shell=shell)

    # Execute
    while True:
        log = proc.stdout.readline().decode().strip()
        err = proc.stderr.readline().decode().strip()

        if log == '' and proc.poll() is not None:
            break

        if log:
            logger.info(prefix + log)

        if err:
            logger.error(prefix + err)


