
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.model import ModelError
import c_hymod_models_gr4j

# Dimensions
nuhmaxlength = c_hymod_models_gr4j.uh_getnuhmaxlength()

nstates = c_hymod_models_gr4j.gr4j_getnstates()

noutputs = c_hymod_models_gr4j.gr4j_getnoutputs()


class GR4J(Model):

    def __init__(self):

        Model.__init__(self, 'gr4j', \
            nuhmaxlength, nstates, 4, 4, \
            ['Q[mm/d]', 'Ech[mm/d]', 'E[mm/d]', 'Pr[mm/d]',\
                'Qd[mm/d]', 'Qr[mm/d]', 'Perc[mm/d]',\
                'S[mm]', 'R[mm]'], \
            [5.7, -0.88, 4.4, -5], \
            [[3, 0, 0, 0], [0, 3, 0, 0],\
                [0, 0, 3, 0], [0, 0, 0, 1]], \
            [10, -50, 10, 0.5], \
            [20000, 50, 5000, 200], \
            [400, -1, 50, 0.5])

        self.nuh1 = 0
        self.nuh2 = 0


    def set_uhparams(self):
        # First uh
        nuh1 = np.zeros(1).astype(np.int32)
        uh1 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models_gr4j.uh_getuh(nuhmaxlength,
                1, self.trueparams[3], \
                nuh1, uh1)
        self.nuh1 = nuh1[0]

        if ierr > 0:
            moderr = ModelError(self.name, ierr, \
                    'exception raised by gr4j_getuh')
            moderr.set_ierr_id()
            raise moderr
            raise GR4JException('gr4j_getuh raised the exception %d' % ierr)
        
        self.uh[:self.nuh1] = uh1[:self.nuh1]

        # Second uh
        nuh2 = np.zeros(1).astype(np.int32)
        uh2 = np.zeros(self.nuhmaxlength).astype(np.float64)
        ierr = c_hymod_models_gr4j.uh_getuh(nuhmaxlength, \
                2, self.trueparams[3], \
                nuh2, uh2)
        self.nuh2 = nuh2[0]

        if ierr > 0:
            moderr = ModelError(self.name, ierr, \
                    'exception raised by gr4j_getuh')
            moderr.set_ierr_id()
            raise moderr
        
        if self.nuh1 + self.nuh2 > self.nuhmaxlength:
            moderr = ModelError(self.name, -1, 'gr4j.set_uhparams')
            moderr.set_ierr('ESIZE_STATESUH')
            raise moderr

        self.uh[self.nuh1:self.nuh1+self.nuh2] = uh2[:self.nuh2]
        self.nuhlength = self.nuh1 + self.nuh2


    def initialise(self, states=None, statesuh=None):

        # initialise GR4J with reservoir levels
        if states is None:
            states = [self.trueparams[0]/2, self.trueparams[2]/2] \
                    + [0] * (self.nstates-2)

        if statesuh is None:
            statesuh = [0.] * self.nuhmaxlength

        states = np.atleast_1d(states).astype(np.float64)
        self.states[:self.nstates] = states[:self.nstates]

        statesuh = np.array(statesuh).astype(np.float64)
        self.statesuh[:self.nuhlength] = statesuh[:self.nuhlength]
        

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
            moderr = ModelError(self.name, ierr, 'c_hymod_models_gr4j.gr4j_run')
            moderr.set_ierr_id()
            raise moderr


    def cal2true(self):
        xt = self.calparams

        self.trueparams = np.array([np.exp(xt[0]), 
                np.sinh(xt[1]),
                np.exp(xt[2]), 
                0.49+np.exp(xt[3])])

