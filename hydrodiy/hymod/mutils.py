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


def sinmodel(params, secofyear):
    ''' Sinusoidal model to produce seasonal daily data '''
    
    # location
    mu = params[0]

    # scale
    eta = params[1]

    # phase
    phi = 2*math.pi * max(0., 
            min(1, math.exp(params[2])/(1+math.exp(params[2]))))

    # shape
    alpha = max(-50, min(50, params[3]))

    # Sinusoid
    nsec = 365.2425 * 86400
    u = np.sin((0. + secofyear)/nsec*2*math.pi + math.pi/2 - phi)

    # Shape power factor
    if abs(alpha) > 1e-3:
        y = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))

    else:
        y = (u+1)/2+alpha/2*(u**2-1)

    return mu + math.exp(eta) * y

