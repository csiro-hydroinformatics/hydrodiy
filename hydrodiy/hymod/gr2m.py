
import numpy as np
import pandas as pd

from hystat import sutils

import c_hymod

class GR2MException(Exception):
    pass

class GR2MSizeException(Exception):
    pass

# Error message number
esize = c_hymod.hymod_getesize()

# Dimensions
nstates = c_hymod.gr2m_getnstates()

noutputs = c_hymod.gr2m_getnoutputs()


class GR2M():

    def __init__(self):
        pass

    def __str__(self):
        str = '\nGR2M model implementation\n'
        str += '  params = [%0.1f, %0.1f]\n' % (
                self.params[0], self.params[1])
        str += '  states = [%0.1f, %0.1f]\n' % (self.statesini[0], 
                self.statesini[1])
        str += '  nout = %d\n' % self.nout

        return str


    def setoutputs(self, nval, nout):
        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)
        self.statesini = np.zeros(nstates).astype(np.float64)

    def setparams(self, params):
        # Set params value
        self.params = np.array(params).astype(np.float64)

    def setstates(self, statesini=None, statesuhini=None):
        if statesini is None:
            statesini = [self.params[0]/2, self.params[1]/2]

        ns = len(statesini)
        self.statesini[:ns] = np.array(statesini).astype(np.float64)

    def run(self, inputs):
        ierr = c_hymod.gr2m_run(self.params, inputs, 
            self.statesini,
            self.outputs)

        if ierr == esize:
            raise GR2MKernelSizeException(('gr2m_run returns a '
                'size exception %d') % ierr)
        if ierr > 0:
            raise GR2MKernelException(('gr2m_run returns an '
                'exception %d') % ierr)

    
    def calib(self, inputs, outputs):
        pass


def get_paramslib(nsamples, seed=333):
    
    np.random.seed(seed)
    
    pmin = [4.5, 0.8]
    pmax = [7.5, 1.2] 
    samples = sutils.lhs(4, nsamples, pmin, pmax)

    samples[:,0] = np.exp(samples[:,0])

    return samples
    
