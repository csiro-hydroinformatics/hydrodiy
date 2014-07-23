import datetime
import numpy as np

def turc_mezentsev(n, P, PE):
    ''' Return runoff computed with the Turc-Mezentsev model  
        Usually n=2
    '''
    return P*(1.-1./(1.+(P/PE)**float(n))**(1./float(n)))

