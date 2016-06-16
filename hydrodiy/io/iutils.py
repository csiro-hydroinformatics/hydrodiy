import sys
import os
import re
import gzip
import datetime

import numpy as np

def password(length=10, chr_start=35, chr_end=128):
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
    >>> pwd = iutils.password()
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

def extracpat(string, regexp):
    '''
        Returns the first hit of a compiled regexp
        regexp should be compiled with re.compile first
    '''

    out = 'NA'
    se = regexp.search(string)
    if se:
        try:
            out = se.group(0)
        except IndexError:
            pass
    return out


def write_var(data):
    ''' Write a simple dict in the format v1[value1]_v2[value2]

    Parameters
    -----------
    data : dict
        Non nested dictionary containing data

    Example
    -----------
    >>> iutils.write_var({'name':'bob', 'phone':2010})

    '''
    out = []
    for k, v in data.iteritems():
        out.append('{0}[{1}]'.format(k, v))

    return '_'.join(out)


def find_var(source):
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
    >>> iutils.find_match(source)

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
                    'on %s\n') % datetime.datetime.now()]
    meta += ['#\n', '# ------------------------------\n']

    txt = txt[:2] + meta + txt[3:]

    with open(filename, 'w') as fs:
        fs.writelines(txt)

