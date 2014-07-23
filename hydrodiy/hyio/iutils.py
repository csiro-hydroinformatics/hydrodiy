import sys
import os
import re


def find_files(folder, pattern):
    ''' Find files recursively based on regexp pattern search '''

    found = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            fn = os.path.join(root, filename)

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

