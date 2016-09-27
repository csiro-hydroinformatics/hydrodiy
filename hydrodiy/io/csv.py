import sys, os, re, time, string

import gzip
import tarfile
import tempfile

# Python 2/3 string io import
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

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
            val = s[len(k)+1:].strip()
            k = re.sub(' +', '_', k.strip().lower())

            if not bool(re.search('\:', s[:key_length_max])):
                k = 'comment_{0:02d}'.format(i)
                val = s
                i += 1

            if val != '':
                comment[k] = val

    return comment


def _csvhead(nrow, ncol, comment, source_file, author=None):
    """ Produces a nice header for csv files """

    # Generate the comments dict
    if isinstance(comment, str):
        comments = {'comment': comment}

    elif isinstance(comment, list):
        comments = {}
        for i in range(len(comment)):
            comments['comment%2.2d' % i] = comment[i]

    elif isinstance(comment, dict):
        comments = {}
        for k in comment:
            comments[re.sub('\:', '', k).lower()] = comment[k]

    else:
        comment = list(comment)

        comments = {}
        for i in range(len(comment)):
            comments['comment%2.2d' % i] = comment[i]


    # Generate file header
    h = []
    h.append("# --------------------------------------------------")

    h.append("# nrow : %d" % nrow)
    h.append("# ncol : %d" % ncol)

    for k in comments:
        h.append("# %s : %s" %(k, comments[k]))

    h.append("# time_generated : %s" % time.strftime("%Y-%m-%d %H:%M"))

    # seek author
    if author is None:
        try:
            author = os.getlogin()
        except:
            author = 'unknown'

    h.append("# author : %s" % author)

    # seek source file
    h.append("# source_file : %s" % source_file)

    # Python config
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


def write_csv(data, filename, comment,
        source_file,
        author=None,
        write_index=False,
        compress=True,
        float_format='%0.3f',
        archive=None,
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
    >>> # Create an empty script
    >>> fo = open('script.py', 'w'); fo.close()
    >>> # Store data
    >>> csv.write_csv(df, 'data.csv', 'This is a test', 'script.py')

    '''

    head = _csvhead(data.shape[0], data.shape[1],
                comment,
                source_file = source_file,
                author=author)

    # Check source_file exists
    if not os.path.exists(source_file):
        raise ValueError('%s file does not exists' % source_file)

    if not archive is None:
        compress = False

    # defines file name
    filename_full = filename

    if compress and ~filename.endswith('.gz'):
        filename_full += '.gz'

    # Open pipe depending on file type
    if archive is None:
        if compress:
            fcsv = gzip.open(filename_full, 'wb')
        else:
            fcsv = open(filename_full, 'w')
    else:
        fcsv = tempfile.NamedTemporaryFile('w', delete=False)
        filename_full = fcsv.name

    # Write header
    for line in head:
        fcsv.write(line+'\n')

    # Write data itself
    data.to_csv(fcsv, index=write_index,
        float_format=float_format, \
        **kwargs)

    fcsv.close()

    if not archive is None:
        # Add file to archive
        archive.add(filename_full, arcname=filename)

        # Delete file
        os.remove(filename_full)




def read_csv(filename, has_colnames=True, archive=None, **kwargs):
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
        # Add gz if file does not exists
        try:
            if not os.path.exists(filename):
                filename += '.gz'

            # Open proper file type
            if filename.endswith('gz'):
                fcsv = gzip.open(filename, 'rb')
            else:
                fcsv = open(filename, 'r')

        except TypeError:
            # Assume filename is stream
            fcsv = filename
    else:
        # Use the archive mode
        fcsv = archive.extractfile(filename)

    # Reads content
    header = []
    comment = {}

    if has_colnames:

        line = fcsv.readline()

        while line.startswith('#'):
            header.append(re.sub('^# *|\n$', '', line))
            line = fcsv.readline()

        # Extract comment info from header
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

