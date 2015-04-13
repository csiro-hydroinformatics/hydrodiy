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


def sinmodel(params, dayofyear):
    ''' Sinusoidal model to produce seasonal daily data '''
    
    mu = params[0] # location
    eta = max(0., params[1]) # scale
    phi = max(0., min(2*np.pi, params[2])) # phase 
    alpha = max(-20, min(20, params[3])) # shape

    # Sinusoid
    u = 0.5 + 0.5*np.sin((0. + dayofyear)/365*2*np.pi + np.pi/2 - phi)

    # Shape power factor
    if np.abs(alpha+1) > 1e-10:
        y = ((1+u)**(alpha+1)-1)/(2**(alpha+1)-1) - 0.5
    else:
        y = (math.log(1+u)-1)/(math.log(2)-1) - 0.5

    return mu + eta * y
