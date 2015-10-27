
import math
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError

import c_hymod_models_gr2m
import c_hymod_models_utils


# Dimensions
nstates = c_hymod_models_gr2m.gr2m_getnstates()

noutputs = c_hymod_models_gr2m.gr2m_getnoutputs()


class GR2M(Model):

    def __init__(self):

        Model.__init__(self, 'gr2m',
            0, \
            0, \
            nstates, \
            2, \
            2, \
            ['S', 'IGF'], \
            ['Q[mm/m]', 'Ech[mm/m]', \
                'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]', \
                'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]'],
            [5.8, -0.3], \
            [[0.6, 0.06], [0.06, 0.08]], \
            [10, 0.1], \
            [10000, 3], \
            [400, 0.8])


    def run(self, inputs):
        ierr = c_hymod_models_gr2m.gr2m_run(self.trueparams, inputs, \
            self.states, \
            self.outputs)

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='returned from c_hymod_models_gr2m.gr2m_run')


    def cal2true(self, calparams):
        return np.array([math.exp(calparams[0]), math.exp(calparams[1])])


