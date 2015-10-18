
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError

import c_hymod_models_dummy

# Dimensions
nstates = c_hymod_models_dummy.dummy_getnstates()

noutputs = c_hymod_models_dummy.dummy_getnoutputs()


class Dummy(Model):

    def __init__(self):

        Model.__init__(self, 
            'dummy', \
            0, nstates, 1, 1, \
            ['o1', 'o2'],
            [-2], \
            [2], \
            [0], \
            [[0.5]], \
            [0])


    def run(self, inputs):

        ierr = c_hymod_models_dummy.dummy_run(self.trueparams, \
            inputs, \
            self.states, \
            self.outputs)

        if ierr > 0:
            moderr = ModelError(self.name, ierr, 
                    'c_hymod_models_dummy.dummy_run')
            moderr.set_ierr_id()
            raise moderr


    def cal2true(self):
        xt = self.calparams
        self.trueparams = np.array([np.exp(xt[0])])

