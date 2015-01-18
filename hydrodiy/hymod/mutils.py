import datetime
import os
import numpy as np

import sh

def turc_mezentsev(n, P, PE):
    ''' Returns runoff computed with the Turc-Mezentsev model  
        Usually n=2.5
    '''
    QT = P*(1.-1./(1.+(P/PE)**float(n))**(1./float(n)))

    return QT

def getmemusage():
    ''' Returns memory usage of current python process 

        Code from http://stackoverflow.com/questions/938733/total-memory-used-by-python-process 
    '''

    mem = float(sh.awk(sh.ps('u','-p',os.getpid()),'{sum=sum+$6}; END {print sum/1024}'))

    return mem
