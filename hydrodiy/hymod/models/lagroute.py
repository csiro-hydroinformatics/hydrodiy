
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError

import c_hymod_models_lagroute
import c_hymod_models_utils

# Dimensions
nuhmaxlength = c_hymod_models_utils.uh_getnuhmaxlength()

nstates = c_hymod_models_lagroute.lagroute_getnstates()

noutputs = c_hymod_models_lagroute.lagroute_getnoutputs()


class LagRoute(Model):

    def __init__(self):

        Model.__init__(self, 'lagroute', \
            4, \
            ['timestep[s]', 'length[m]', 'flowref[m3/s]', 'storageexpon[-]'], \
            [86400, 1e5, 1, 1], \
            1, 
            nuhmaxlength, \
            nstates, \
            2, \
            2, \
            ['U', 'alpha'], \
            ['Q[m3/s]', 'laggedQ[m3/s]', 'Vlag[m3]', 'Vstore[m3]'], \
            [1., 0.5], \
            [[0.2, 0], [0, 0.1]],\
            [0.01, 0.], \
            [20., 1.], \
            [1, 0.5])


    def set_uhparams(self):

        # Lag = alpha * U * L / dt
        delta = self.config[1] * self.trueparams[0] * self.trueparams[1]
        delta /= self.config[0]
        delta = np.float64(delta)

        # First uh
        nuh = np.zeros(1).astype(np.int32)
        uh = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(nuhmaxlength,
                5, delta, \
                nuh, uh)

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='c_hymod_models_utils.uh_getuh')

        self.uh = uh
        self.nuhlength = nuh[0]


    def run(self, inputs):

        if inputs.shape[1] != self.ninputs:
            raise ModelError(self.name, 
                    ierr_id='ESIZE_INPUTS', \
                    message='returned from LagRoute.run')

        ierr = c_hymod_models_lagroute.lagroute_run(self.nuhlength, \
            self.config, \
            self.trueparams, \
            self.uh, \
            inputs, \
            self.statesuh, \
            self.states, \
            self.outputs)

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                message='c_hymod_models_lagroute.lagroute_run')


