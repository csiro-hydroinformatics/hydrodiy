import sys
import os
import re
import gzip
import datetime


def find_files(folder, pattern, recursive=True):
    ''' Find files recursively based on regexp pattern search '''

    found = []

    if recursive:
        for root, dirs, files in os.walk(folder):
            for filename in files:
                fn = os.path.join(root, filename)

                if not re.search(pattern, fn) is None:
                    found.append(fn)
    else:
        files = next(os.walk(folder))[2]
        found = [os.path.join(folder, f) for f in files]

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

def script_template(filename, author='J. Lerat, EHP, Bureau of Meteorogoloy'):
    '''
        Write a script template into a text file
    '''

    FMOD, modfile = os.path.split(__file__)
    f = os.path.join(FMOD, 'script_template.py.gz')
    with gzip.GzipFile(f, 'rb') as fgz:
        txt = fgz.readlines()

    meta = ['# -- Script Meta Data --\n']
    meta += ['# Author : %s\n' % author]
    meta += ['# Versions :\n']
    meta += ['#    V00 - Script written from template on %s\n' % datetime.datetime.now()]
    meta += ['#\n', '# ------------------------------\n']

    txt = txt[:2] + meta + txt[3:]
    
    with open(filename, 'w') as fs:
        fs.writelines(txt)

        
    

