import datetime
import math 
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


def sinmodel(params, secofyear):
    ''' Sinusoidal model to produce seasonal daily data '''
    
    mu = params[0] # location
    eta = max(0., params[1]) # scale
    phi = max(0., min(2*np.pi, params[2])) # phase 
    alpha = max(-50, min(50, params[3])) # shape

    # Sinusoid
    nsec = 365.2425 * 86400
    u = np.sin((0. + secofyear)/nsec*2*np.pi + np.pi/2 - phi)

    # Shape power factor
    if abs(alpha) > 1e-3:
        y = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))

    else:
        y = (u+1)/2+alpha/2*(u**2-1)

    return mu + eta * y

