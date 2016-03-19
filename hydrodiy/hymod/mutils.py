import datetime
import math
import os
import numpy as np

import sh

def getmemusage():
    ''' Returns memory usage of current python process

        Code from http://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    '''

    mem = float(sh.awk(sh.ps('u','-p',os.getpid()),
            '{sum=sum+$6}; END {print sum/1024}'))

    return mem

