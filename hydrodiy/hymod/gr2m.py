
import numpy as np
import pandas as pd

from hystat import sutils

from model import Model
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


class GR2M(Model):

    def __init__(self):

        Model.__init__(self, 'gr2m', 0, nstates, \
            2, 2, \
            ['Q[mm/m]', 'Ech[mm/m]', \
                'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]', \
                'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]'],
            [1, -2], \
            [20, 1], \
            [5.7, -0.2], \
            [[3, 0], [0, 1]])


    def run(self, inputs):
        ierr = c_hymod.gr2m_run(self.trueparams, inputs, 
            self.statesini,
            self.outputs)

        if ierr == esize:
            raise GR2MKernelSizeException(('gr2m_run returns a '
                'size exception %d') % ierr)
        if ierr > 0:
            raise GR2MKernelException(('gr2m_run returns an '
                'exception %d') % ierr)


    def cal2true(self):
        x = self.calparams
        self.trueparams = np.array([np.exp(xt[0]), np.exp(xt[1])])


