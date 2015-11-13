
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration

import c_hymod_models_lagroute
import c_hymod_models_utils

# Dimensions
NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()


class LagRoute(Model):

    def __init__(self):

        Model.__init__(self, 'lagroute',
            nconfig=4, \
            ninputs=1, \
            nparams=2, \
            nstates=1, \
            noutputs_max = 4,
            inputs_names = ['Inflow'], \
            outputs_names = ['Q[m3/s]', 'laggedQ[m3/s]', \
                'Vlag[m3]', 'Vstore[m3]'])

        self.config.names = ['timestep', 'length', \
                'flowref', 'storage_expon']
        self.config.units = ['s', 'm', 'm3/s', '-']

        self.states.names = ['V']
        self.states.units = ['m3']

        self.params.names = ['U', 'alpha']
        self.params.units = ['s/m', '-']
        self.params.min = [0.01, 0.]
        self.params.max = [20., 1.]
        self.params.default = [1., 0.5]


    def set_uh(self):

        # Lag = alpha * U * L / dt
        config = self.config
        params = self.params

        delta = config['length'] * params['U'] * params['alpha']
        delta /= self.config['timestep']
        delta = np.float64(delta)

        if np.isnan(delta):
            raise ValueError(('Problem with delta calculation. ' + \
                'One of config[\'length\']{0}, config[\'timestep\']{1}, ' + \
                'params[\'U\']{2} or params[\'alpha\']{3} is NaN').format( \
                config['length'], config['timestep'], params['U'],
                params['alpha']))

        # First uh
        nuh = np.zeros(1).astype(np.int32)
        uh = np.zeros(NUHMAXLENGTH).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(NUHMAXLENGTH,
                5, delta, \
                nuh, uh)

        if ierr > 0:
            raise ValueError(('Model LagRoute: c_hymod_models_utils.uh_getuh' + \
                ' returns {0}').format(ierr))

        self._uh.data = uh
        self._nuhlength = nuh[0]


    def run(self):

        if self.inputs.nvar != self.ninputs:
            raise ValueError(('Model LagRoute, self.inputs.nvar({0}) != ' + \
                    'self.ninputs({1})').format( \
                    self._inputs.nvar, self._ninputs))

        ierr = c_hymod_models_lagroute.lagroute_run(self._nuhlength, \
            self.config.data, \
            self.params.data, \
            self.uh.data, \
            self.inputs.data, \
            self.statesuh.data, \
            self.states.data, \
            self.outputs.data)

        if ierr > 0:
            raise ValueError(('c_hymod_models_lagroute.' + \
                'lagroute_run returns {0}').format(ierr))


class CalibrationLagRoute(Calibration):

    def __init__(self, timeit=False):

        lm = LagRoute()

        Calibration.__init__(self, 
            model = lm, \
            ncalparams = 2, \
            timeit = timeit)

        self.calparams_means.data =  [1., 0.5]

        stdevs = [0.5, 0., 0., 0.2]
        self.calparams_stdevs.data = stdevs


