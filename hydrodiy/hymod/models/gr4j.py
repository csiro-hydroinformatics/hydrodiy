
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError

import c_hymod_models_gr4j
import c_hymod_models_utils

# Dimensions
nuhmaxlength = c_hymod_models_utils.uh_getnuhmaxlength()

nstates = c_hymod_models_gr4j.gr4j_getnstates()

noutputs = c_hymod_models_gr4j.gr4j_getnoutputs()


class GR4J(Model):

    def __init__(self):

        self.nuh1 = 0
        self.nuh2 = 0

        Model.__init__(self, 'gr4j', \
            0, \
            nuhmaxlength, \
            nstates, \
            4, \
            4, \
            ['S', 'IGF', 'R', 'TB'], \
            ['Q[mm/d]', 'Ech[mm/d]', 'E[mm/d]', 'Pr[mm/d]',\
                'Qd[mm/d]', 'Qr[mm/d]', 'Perc[mm/d]',\
                'S[mm]', 'R[mm]'], \
            [5.7, -0.88, 4.4, -1], \
            [[3, 0, 0, 0], [0, 3, 0, 0],\
                [0, 0, 3, 0], [0, 0, 0, 1]], \
            [10, -50, 1, 0.5], \
            [20000, 50, 5000, 100], \
            [400, -1, 50, 0.5])


    def set_uhparams(self):

        # First uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(nuhmaxlength,
                1, self.trueparams[3], \
                nuh1, uh1)
        self.nuh1 = nuh1[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='c_hymod_models_utils.uh_getuh')

        self.uh[:self.nuh1] = uh1[:self.nuh1]

        # Second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models_utils.uh_getuh(nuhmaxlength, \
                2, self.trueparams[3], \
                nuh2, uh2)
        self.nuh2 = nuh2[0]

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                    message='c_hymod_models_utils.uh_getuh')

        if self.nuh1 + self.nuh2 > self.nuhmaxlength:
            raise ModelError(self.name, \
                ierr_id = 'ESIZE_STATESUH', \
                message='gr4j.set_uhparams')

        self.uh[self.nuh1:self.nuh1+self.nuh2] = uh2[:self.nuh2]
        self.nuhlength = self.nuh1 + self.nuh2


    def initialise(self, states=None, statesuh=None):

        # initialise GR4J with reservoir levels
        if states is None:
            states = np.zeros(self.nstates)
            states[0] = self.trueparams[0] * 0.5
            states[1] = self.trueparams[2] * 0.4

        super(GR4J, self).initialise(states, statesuh)


    def run(self, inputs):

        ierr = c_hymod_models_gr4j.gr4j_run(self.nuh1, \
            self.nuh2, \
            self.trueparams, \
            self.uh, \
            self.uh[self.nuh1:], \
            inputs, \
            self.statesuh, \
            self.states, \
            self.outputs)

        if ierr > 0:
            raise ModelError(self.name, ierr, \
                message='c_hymod_models_gr4j.gr4j_run')


    def cal2true(self, calparams):
        trueparams = np.array([np.exp(calparams[0]),
                np.sinh(calparams[1]),
                np.exp(calparams[2]),
                0.49+np.exp(calparams[3])])

        return trueparams

