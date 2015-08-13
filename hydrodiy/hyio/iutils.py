import sys
import os
import re
import gzip
import datetime


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

    Example
    -----------
    Look for all python scripts in current folder
    >>> lf = iutils.find_files('.', '.*\\.py', Fase)
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

