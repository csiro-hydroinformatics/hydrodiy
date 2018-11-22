''' Module to read and write csv data with comments '''

import sys, os, re, json
import zipfile

from hydrodiy import PYVERSION

# Tailor string handling depending on python version
if PYVERSION == 2:
    from StringIO import StringIO
    UNICODE = unicode

elif PYVERSION == 3:
    from io import StringIO
    UNICODE = str


def write_zipjson(data, filename, compression=zipfile.ZIP_DEFLATED, **kwargs):
    ''' write a python object to a compressed json file

    Parameters
    -----------
    data : object
        Serializable python object
    filename : str
        Path to file
    compression : str
        Compression level
    kwargs : dict
        Arguments passed to json.dump
    '''
    # Write to Json
    filename_txt = re.sub('\.[^\.]+$', '.json', filename)
    with open(filename_txt, 'w') as fo:
        json.dump(data, fo, **kwargs)

    # Zip the file
    with zipfile.ZipFile(filename, 'w', compression=compression) as archive:
        archive.write(filename_txt, arcname=os.path.basename(filename_txt))

    # Remove file
    os.remove(filename_txt)


def read_zipjson(filename, encoding='utf-8', **kwargs):
    ''' Read a json zipped file

    Parameters
    -----------
    filename : str
        Path to file
    encoding : str
        Charactert encoding
    kwargs : dict
        Arguments passed to json.load
    '''
    data = {}
    nfiles = 0
    with zipfile.ZipFile(filename, 'r') as archive:
        for fname in archive.namelist():
            uni = UNICODE(archive.read(fname), encoding=encoding)
            with StringIO(uni) as fobj:
                data[fname] = json.load(fobj, **kwargs)

            nfiles += 1

    # Simplify output if there is one file only
    if nfiles == 1:
        data = data[fname]

    return data

