
import numpy as np
import pandas as pd

from hystat import sutils

from model import Model
import c_hymod

class GR4JException(Exception):
    pass

class GR4JSizeException(Exception):
    pass

# Error message number
esize = c_hymod.hymod_getesize()

# Dimensions
nuhmax = c_hymod.gr4j_getnuhmax()

nstates = c_hymod.gr4j_getnstates()

noutputs = c_hymod.gr4j_getnoutputs()


class GR4J(Model):

    def __init__(self):

        Model.__init__(self, 'gr4j', nuhmax, nstates, \
            4, 4, \
            ['Q[mm/d]', 'Ech[mm/d]', 'E[mm/d]', 'Pr[mm/d]',\
                'Qd[mm/d]', 'Qr[mm/d]', 'Perc[mm/d]',\
                'S[mm]', 'R[mm]'], \
            [1, -5, 1, -10], \
            [20, 5, 9, 4], \
            [5.7, -0.88, 4.4, -5], \
            [[3, 0, 0, 0], [0, 3, 0, 0],\
                [0, 0, 3, 0], [0, 0, 0, 1]])


    def set_uhparams(self):
        nuh_optimised = np.zeros(2).astype(np.int32)

        ierr = c_hymod.gr4j_getuh(self.trueparams[3], nuh_optimised, self.uh)
        if ierr > 0:
            raise GR4JException('gr4j_getuh raised the exception %d' % ierr)

        self.nuh = nuh_optimised[0]

    def run(self, inputs):
        ierr = c_hymod.gr4j_run(self.nuh, \
            self.trueparams, self.uh, \
            inputs, \
            self.statesuh, \
            self.states, \
            self.outputs)

        if ierr == esize:
            raise GR4JKernelSizeException(('gr4j_run returns a '
                'size exception %d') % ierr)
        if ierr > 0:
            raise GR4JKernelException(('gr4j_run returns an '
                'exception %d') % ierr)

    def cal2true(self):
        xt = self.calparams
        self.trueparams = np.array([np.exp(xt[0]), np.sinh(xt[1]),
                np.exp(xt[2]), 0.5+np.exp(xt[3])])

