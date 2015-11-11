
import math
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration

import c_hymod_models_gr2m
import c_hymod_models_utils


class GR2M(Model):

    def __init__(self):

        Model.__init__(self, 'gr2m',
            nconfig=1, \
            ninputs=2, \
            nparams=2, \
            nstates=2, \
            noutputs_max = 9,
            inputs_names = ['P', 'PET'], \
            outputs_names = ['Q[mm/m]', 'Ech[mm/m]', \
                'P1[mm/m]', 'P2[mm/m]', 'P3[mm/m]', \
                'R1[mm/m]', 'R2[mm/m]', 'S[mm]', 'R[mm]'])

        self.config.names = 'catcharea'
        
        self.states.names = ['Sr', 'Rr']
        self.states.units = ['mm', 'mm']

        self.params.names = ['S', 'IGF']
        self.params.units = ['mm', '-']
        self.params.min = [10., 0.1]
        self.params.max = [10000., 3.]
        self.params.default = [400., 0.8]


    def run(self):

        ierr = c_hymod_models_gr2m.gr2m_run(self.params.data, \
            self.inputs.data, \
            self.states.data, \
            self.outputs.data)

        if ierr > 0:
            raise ValueError(('Model gr2m, c_hymod_models_gr2m.gr2m_run' + \
                    'returns {0}').format(ierr))

    def cal2true(self, calparams):
        return np.exp(calparams)



class CalibrationGR2M(Calibration):

    def __init__(self, observations, inputs, timeit=False):

        gr = GR2M()

        Calibration.__init__(self, 
            model = gr, \
            ncalparams = 2, \
            observations = observations, \
            timeit = timeit)

        self.calparams_means.data =  [5.8, -0.2]
        self.calparams_stdevs.data = [1., 0., 0., 0.1]

        self._model.allocate(self._observations.nval, 1)
        self._model.inputs.data = inputs


    def cal2true(self, calparams):
        return np.exp(calparams)


