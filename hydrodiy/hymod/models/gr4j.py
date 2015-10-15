
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
import c_hymod_models

class GR4JException(Exception):
    pass

class GR4JSizeException(Exception):
    pass

# Error message number
esize = c_hymod_models.getesize()

# Dimensions
nuhmaxlength = c_hymod_models.uh_getnuhmaxlength()

nstates = c_hymod_models.gr4j_getnstates()

noutputs = c_hymod_models.gr4j_getnoutputs()


class GR4J(Model):

    def __init__(self):

        Model.__init__(self, 'gr4j', \
            nuhmaxlength, nstates, 4, 4, \
            ['Q[mm/d]', 'Ech[mm/d]', 'E[mm/d]', 'Pr[mm/d]',\
                'Qd[mm/d]', 'Qr[mm/d]', 'Perc[mm/d]',\
                'S[mm]', 'R[mm]'], \
            [1, -5, 1, -10], \
            [20, 5, 9, 4], \
            [5.7, -0.88, 4.4, -5], \
            [[3, 0, 0, 0], [0, 3, 0, 0],\
                [0, 0, 3, 0], [0, 0, 0, 1]])

        self.nuh1 = 0
        self.nuh2 = 0


    def set_uhparams(self):
        # First uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models.uh_getuh(1, self.trueparams[3], \
                nuh1, uh1)
        self.nuh1 = nuh1[0]

        if ierr > 0:
            raise GR4JException('gr4j_getuh raised the exception %d' % ierr)
        
        self.uh[:self.nuh1] = uh1[:self.nuh1]

        # Second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models.uh_getuh(2, self.trueparams[3], \
                nuh2, uh2)
        self.nuh2 = nuh2[0]

        if ierr > 0:
            raise GR4JException('gr4j_getuh raised the exception %d' % ierr)
        
        if self.nuh1 + self.nuh2 > self.nuhmaxlength:
            raise GR4JException('nuh1(%d)+nuh2(%d) > nuhmaxlength(%d)' % ( \
                    self.nuh1, self.nuh2, self.nuhmaxlength))

        self.uh[self.nuh1:self.nuh1+self.nuh2] = uh2[:self.nuh2]
        self.nuhlength = self.nuh1 + self.nuh2


    def run(self, inputs):

        ierr = c_hymod_models.gr4j_run(self.nuh1, self.nuh2, \
            self.trueparams, \
            self.uh, \
            self.uh[self.nuh1:], \
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

