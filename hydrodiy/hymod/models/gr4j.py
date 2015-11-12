
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration

import c_hymod_models_gr4j
import c_hymod_models_utils

# Dimensions
NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()


class GR4J(Model):

    def __init__(self):

        self._nuh1 = 0
        self._nuh2 = 0

        Model.__init__(self, 'gr4j',
            nconfig=1, \
            ninputs=2, \
            nparams=4, \
            nstates=2, \
            noutputs_max = 9,
            inputs_names = ['P', 'PET'], \
            outputs_names = ['Q[mm/d]', 'Ech[mm/d]', 'E[mm/d]', 'Pr[mm/d]',\
                'Qd[mm/d]', 'Qr[mm/d]', 'Perc[mm/d]',\
                'S[mm]', 'R[mm]'])

        self.config.names = 'catcharea'
        self.config.units = 'km2'

        self.states.names = ['Sr', 'Rr']
        self.states.units = ['mm', 'mm']

        self.params.names = ['S', 'IGF', 'R', 'TB']
        self.params.units = ['mm', 'mm/d', 'mm', 'd']
        self.params.min = [10., -50., 1., 0.5]
        self.params.max = [20000., 50., 5000., 100.]
        self.params.default = [400., -1., 50., 0.5]

        self.params.reset()


    def set_uh(self):

        params = self.params.data

        # First uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(NUHMAXLENGTH,
                1, params[3], \
                nuh1, uh1)
        self._nuh1 = nuh1[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='c_hymod_models_utils.uh_getuh')

        self.uh.data[:self._nuh1] = uh1[:self._nuh1]

        # Second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(NUHMAXLENGTH, \
                2, params[3], \
                nuh2, uh2)
        self._nuh2 = nuh2[0]

        if ierr > 0:
            raise ValueError('c_hymod_models_utils.uh_getuh returns {0}'.format(\
                ierr))

        nend = self._uh.nval-self._nuh1
        self._uh.data[self._nuh1:] = uh2[:nend]
        self._nuhlength = self._nuh1 + self._nuh2



    def initialise(self, states=None, statesuh=None):
        
        params = self.params.data

        # initialise GR4J with reservoir levels
        if states is None:
            states = np.zeros(self.states.nval)
            states[0] = params[0] * 0.5
            states[1] = params[2] * 0.4

        super(GR4J, self).initialise(states, statesuh)


    def run(self):

        if self.inputs.nvar != self.ninputs:
            raise ValueError(('self.inputs.nvar({0}) != ' + \
                    'self.ninputs({1})').format( \
                    self._inputs.nvar, self._ninputs))
        
        ierr = c_hymod_models_gr4j.gr4j_run(self._nuh1, \
            self._nuh2, \
            self.params.data, \
            self.uh.data, \
            self.uh.data[self._nuh1:], \
            self.inputs.data, \
            self.statesuh.data, \
            self.states.data, \
            self.outputs.data)

        if ierr > 0:
            raise ValueError('c_hymod_models_gr4j.gr4j_run returns {0}'.format(\
                ierr))


class CalibrationGR4J(Calibration):

    def __init__(self, timeit=False):

        gr = GR4J()

        Calibration.__init__(self, 
            model = gr, \
            ncalparams = 4, \
            timeit = timeit)

        self.calparams_means.data =  [5.8, -0.78, 3.39, 0.86]

        stdevs = [1.16, 0.2, -0.15, -0.07, \
                0.2, 1.79, -0.24, -0.149, \
                -0.15, -0.24, 1.68, -0.16, \
                -0.07, -0.149, -0.16, 0.167]
        self.calparams_stdevs.data = stdevs


    def cal2true(self, calparams):
        params = np.array([np.exp(calparams[0]),
                np.sinh(calparams[1]),
                np.exp(calparams[2]),
                0.5+np.exp(calparams[3])])

        return params


