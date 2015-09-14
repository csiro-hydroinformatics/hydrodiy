
import numpy as np
import pandas as pd

from hystat import sutils

import c_hymod

class GR4JException(Exception):
    pass

class GR4JSizeException(Exception):
    pass

# Error message number
esize = c_hymod.hymod_getesize()

# Dimensions
nuh = c_hymod.gr4j_getnuh()

nstates = c_hymod.gr4j_getnstates()

noutputs = c_hymod.gr4j_getnoutputs()


class GR4J():

    def __init__(self):
        self.uh = np.zeros(nuh).astype(np.float64)

    def __str__(self):
        str = '\nGR4J model implementation\n'
        str += '  params = [%0.1f, %0.1f, %0.1f, %0.1f]\n' % (
                self.params[0], self.params[1], self.params[2], self.params[3])
        str += '  states = [%0.1f, %0.1f]\n' % (self.statesini[0], 
                self.statesini[1])
        str += '  nuh = %d\n' % self.nuh
        str += '  nout = %d\n' % self.nout

        return str


    def setoutputs(self, nval, nout):
        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)
        self.uh = np.zeros(nuh).astype(np.float64)
        self.statesini = np.zeros(nstates).astype(np.float64)
        self.statesuhini = np.zeros(nuh).astype(np.float64)


    def setparams(self, params):
        # Set params value
        self.params = np.array(params).astype(np.float64)

        # Set uh
        nuh_optimised = np.zeros(2).astype(np.int32)
        c_hymod.gr4j_getuh(self.params[3], nuh_optimised, self.uh)
        self.nuh = nuh_optimised[0]


    def setstates(self, statesini=None, statesuhini=None):
        if statesini is None:
            statesini = [self.params[0]/2, self.params[2]/2]

        if statesuhini is None:
            statesuhini = [0.] * self.nuh

        ns = len(statesini)
        self.statesini[:ns] = np.array(statesini).astype(np.float64)

        statesuhini = np.array(statesuhini[:self.nuh]).astype(np.float64)
        self.statesuhini[:self.nuh] = statesuhini


    def run(self, inputs):
        ierr = c_hymod.gr4j_run(self.nuh, self.params, self.uh, 
            inputs, 
            self.statesuhini, 
            self.statesini,
            self.outputs)

        if ierr == esize:
            raise GR4JKernelSizeException(('gr4j_run returns a '
                'size exception %d') % ierr)
        if ierr > 0:
            raise GR4JKernelException(('gr4j_run returns an '
                'exception %d') % ierr)

    
    def calib(self, inputs, outputs):
        pass



def transpar_forward(x):
    xt = [np.log(x[0]), np.arcsinh(x[1]),
            np.log(x[2]), np.log(x[3])]

    return xt


def transpar_inverse(xt):
    x = [np.exp(xt[0]), np.sinh(xt[1]),
            np.exp(xt[2]), np.exp(xt[3])]

    return x


def get_paramslib(nsamples, seed=333):
    
    np.random.seed(seed)
    
    pmin = [4.5, -3, 4, 0.5]
    pmax = [7.5, 0, 6, 2] 
    samples = sutils.lhs(4, nsamples, pmin, pmax)

    samples[:,0] = np.exp(samples[:,0])
    samples[:,2] = np.exp(samples[:,2])

    return samples
    
