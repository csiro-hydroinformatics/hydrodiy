
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError

import c_hymod_models_gr2m



# Error message number
esize = c_hymod_models_gr2m.getesize()

# Dimensions
nstates = c_hymod_models_gr2m.gr2m_getnstates()

noutputs = c_hymod_models_gr2m.gr2m_getnoutputs()


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
            [[3, 0], [0, 1]], \
            [400, 0.8])


    def run(self, inputs):
        ierr = c_hymod_models_gr2m.gr2m_run(self.trueparams, inputs, 
            self.states,
            self.outputs)

        if ierr > 0:

            moderr = ModelError(self.name, ierr, 
                    'c_hymod_models_gr2m.gr2m_run')
            moderr.set_ierr_id()
            raise moderr


    def cal2true(self):
        xt = self.calparams
        self.trueparams = np.array([np.exp(xt[0]), np.exp(xt[1])])


