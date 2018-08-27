''' Module to read and write csv data with comments '''

import sys, os, re

from datetime import datetime

import gzip
import zipfile
import pandas as pd
import numpy as np

from hydrodiy import PYVERSION

# Tailor string handling depending on python version
if PYVERSION == 2:
    from StringIO import StringIO
    UNICODE = unicode

elif PYVERSION == 3:
    from io import StringIO
    UNICODE = str

# Check distutils available for python folders
HAS_DISTUTILS = False
try:
    from distutils.sysconfig import get_python_inc
    from distutils.sysconfig import get_python_lib
    HAS_DISTUTILS = True
except ImportError:
    pass

# Max length of comment keys
KEY_LENGTH_MAX = 30


def _header2comment(header):
    ''' Format header data into a comment string '''
    comment = {}
    i = 1

    for elem in header:
        if not bool(re.search('-{10}', elem)):
            key = re.sub(':.*$', '', elem)
            val = elem[len(key)+1:].strip()
            key = re.sub(' +', '_', key.strip().lower())

            if not bool(re.search(':', elem[:KEY_LENGTH_MAX])):
                key = 'comment_{0:02d}'.format(i)
                val = elem
                i += 1

            if val != '':
                comment[key] = val

    return comment


def _csvhead(nrow, ncol, comment, source_file, author=None):
    """ Produces a nice header for csv files """

    # Generate the comments dict
    if isinstance(comment, str):
        comments = {'comment': comment}

    elif isinstance(comment, list):
        comments = {}
        for i, com in enumerate(comment):
            comments['comment%2.2d' % i] = com

    elif isinstance(comment, dict):
        comments = {}
        for k in comment:
            comments[re.sub('\:', '', k).lower()] = comment[k]

    else:
        comment = list(comment)
        comments = {}
        for i, com in enumerate(comment):
            comments['comment%2.2d' % i] = com

    # Generate file header
    head = []
    head.append('# --------------------------------------------------')
    head.append('# nrow : {0}'.format(nrow))
    head.append('# ncol : {0}'.format(ncol))

    for key in sorted(comments):
        head.append('# {0} : {1}'.format(key, comments[key]))

    now = datetime.now()
    head.append('# time_generated : ' + \
        '{0}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(\
            now.year, now.month, now.day, now.hour, now.minute,
            now.second))

    # seek author
    if author is None:
        try:
            author = os.getlogin()
        except Exception:
            author = 'unknown'

    head.append('# author : ' + author)

    # seek source file
    head.append('# source_file : ' + source_file)

    # Python config
    head.append('# work_dir : ' + os.getcwd())
    head.append('# python_environment : ' + os.name)
    head.append('# python_version : ' + sys.version.replace('\n', ' '))
    head.append('# pandas_version : ' + pd.__version__)
    head.append('# numpy_version : ' + np.__version__)

    if HAS_DISTUTILS:
        head.append('# python_inc : ' + get_python_inc())
        head.append('# python_lib : ' + get_python_lib())

    head.append('# --------------------------------------------------')

    return head


def _check_name(filename):
    ''' Check the name and add the relevant extension '''

    filename_full = str(filename)

    if os.path.exists(filename_full):
        return filename_full

    if not os.path.exists(filename_full):
        filename_full = re.sub('csv$', 'gz', filename)

    if not os.path.exists(filename_full):
        filename_full = '{0}.gz'.format(filename)

    if not os.path.exists(filename_full):
        filename_full = re.sub('csv$', 'zip', filename)

    if not os.path.exists(filename_full):
        filename_full = '{0}.zip'.format(filename)

    if not os.path.exists(filename_full):
        raise ValueError('Cannot find file {0} (filename={1})'.format(\
            filename_full, filename))

    return filename_full


def write2zip(archive, arcname, txt):
    ''' Write txt to fcsv file in zip archive '''

    # Check file is not in the archive
    zinfo_test = archive.NameToInfo.get(arcname)
    if not zinfo_test is None:
        raise ValueError('File {0} already exists in archive'.format(arcname))

    # Create fileinfo object
    zinfo = zipfile.ZipInfo(arcname)

    now = datetime.now()
    zinfo.date_time = (now.year, now.month, now.day, \
                            now.hour, now.minute, now.second)
    # File permission
    zinfo.external_attr = 33488896 # 0777 << 16

    # Write to archive with header
    archive.writestr(zinfo, txt, \
        compress_type=zipfile.ZIP_DEFLATED)


def write_csv(data, filename, comment,\
        source_file,\
        author=None,\
        write_index=False,\
        compress=True,\
        float_format='%0.5f',\
        archive=None,\
        **kwargs):
    ''' write a pandas dataframe to csv with comments in header

    Parameters
    -----------
    data : pandas.DataFrame
        Dataframe to be written to file
    filename : str
        Path to file
    comment : str (or dict)
        Comments to be added to header
    source_file : str
        Path to script used to generate the data
    author : str
        Data author (default is given by os.getlogin)
    write_index : bool
        Write dataframe index or not (default not)
    compress : bool
        Compress data to gzip format
    float_format : str
        Floating point number format
    archive : tarfile.TarFile
        Archive to which data is to be added.
    kwargs : dict
        Arguments passed to pd.DataFrame.to_csv

    Example
    -----------
    >>> df = pd.DataFrame(np.random.uniform(0, 1, (100, 4))
    >>> # Create an empty source file
    >>> fo = open('script.py', 'w'); fo.close()
    >>> # Store data
    >>> csv.write_csv(df, 'data.csv', 'This is a test', 'script.py')

    '''
    # Check inputs
    data = pd.DataFrame(data)

    # Generate head
    head = _csvhead(data.shape[0], data.shape[1],\
                comment,\
                source_file=source_file,\
                author=author)

    # Check source_file exists
    if not os.path.exists(source_file):
        raise ValueError(source_file + ' file does not exists')

    if not archive is None:
        compress = False

    # defines file name
    filename_full = filename

    if compress and ~filename.endswith('.zip'):
        filename_full = re.sub('csv$', 'zip', filename)

    # Open pipe depending on file type
    if archive or compress:
        # No pipe, store dataframe in a string
        # and then write it to archive
        fobj = None
    else:
        fobj = open(filename_full, 'w')

        # Write header
        for line in head:
            fobj.write(line+'\n')

    # Write data itself
    txt = data.to_csv(fobj, index=write_index,\
        float_format=float_format, \
        **kwargs)

    # Store compressed data
    if archive or compress:
        txt = '\n'.join(head) + '\n' + txt

        # If compress argument, create a new zip file
        arcname = filename
        if compress:
            arcname = os.path.basename(filename)
            archive = zipfile.ZipFile(filename_full, \
                            mode='w', \
                            compression=zipfile.ZIP_DEFLATED)

        write2zip(archive, arcname, txt)

        if compress:
            archive.close()

    else:
        fobj.close()


def read_csv(filename, has_colnames=True, archive=None, \
        encoding='utf-8', **kwargs):
    ''' Read a pandas dataframe from a csv with comments in header

    Parameters
    -----------
    filename : str
        Path to file
    has_colnames : bool
        Are column names stored in the first line of the data ?
    archive : tarfile.TarFile
        Archive to which data is to be added.
    kwargs : dict
        Arguments passed to pd.read_csv

    Example
    -----------
    >>> df = pd.DataFrame(np.random.uniform(0, 1, (100, 4))
    >>> # Create an empty script
    >>> fo = open('script.py', 'w'); fo.close()
    >>> # Store data
    >>> csv.write_csv(df, 'data.csv', 'This is a test', 'script.py')
    >>> df2 = csv.read_csv('data.csv')

    '''
    if archive is None:
        try:
            # Add gz or zip if file does not exists
            filename_full = _check_name(filename)

            # Open proper file type
            if filename_full.endswith('gz'):
                with gzip.open(filename_full, 'rb') as gzipfile:
                    uni = UNICODE(gzipfile.read(), encoding=encoding)
                    fobj = StringIO(uni)

            elif filename_full.endswith('zip'):
                # Extract the data from archive
                # assumes that file is stored at base level
                with zipfile.ZipFile(filename_full, 'r') as zarchive:
                    fbase = re.sub('zip$', 'csv', os.path.basename(filename))
                    uni = UNICODE(zarchive.read(fbase), encoding=encoding)
                    fobj = StringIO(uni)

            else:
                if PYVERSION == 3:
                    fobj = open(filename_full, 'r', encoding=encoding)
                elif PYVERSION == 2:
                    fobj = open(filename_full, 'r')

        except TypeError:
            import warnings
            warnings.warn(('Failed to open {0} as a text file. '+\
                'Will now try to open it as a buffer').format(filename), \
                stacklevel=2)
            fobj = filename

    else:
        # Use the archive mode
        uni = UNICODE(archive.read(filename), encoding=encoding)
        fobj = StringIO(uni)

    # Check fobj is readable
    if not hasattr(fobj, 'readline'):
        raise ValueError('File object is not readable')

    # Reads content
    header = []
    comment = {}

    if has_colnames:
        line = fobj.readline()

        while line.startswith('#'):
            header.append(re.sub('^# *|\n$', '', line))
            line = fobj.readline()

        # Extract comment info from header
        comment = _header2comment(header)

        # Generate column headers
        if 'names' not in kwargs:
            # deals with multi-index columns
            # reformat columns like (idx1-idx2-...)
            search = re.findall('\"\([^\(]*\)\"', line)
            linecols = line

            if search:
                for cn in search:
                    cn2 = re.sub('\(|\)|\"|\'', '', cn)
                    cn2 = re.sub(', *', '-', cn2)
                    linecols = re.sub(re.escape(cn), cn2, linecols)

            cns = linecols.strip().split(',')
        else:
            cns = kwargs['names']
            kwargs.pop('names')

        # Reads data with proper column names
        data = pd.read_csv(fobj, names=cns, \
                    encoding=encoding, **kwargs)

        data.columns = [re.sub('\\.', '_', cn)\
                                        for cn in data.columns]

    else:
        # Reads data frame without header
        data = pd.read_csv(fobj, header=None, \
                    encoding=encoding, **kwargs)

    fobj.close()

    return data, comment

