
import math
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration


class TurcMezentsev(Model):

    def __init__(self):

        Model.__init__(self, 'turcmezentsev',
            nconfig=1, \
            ninputs=2, \
            nparams=1, \
            nstates=1, \
            noutputs_max = 2,
            inputs_names = ['P', 'PET'], \
            outputs_names = ['Q[mm/y]', 'E[mm/y]'])

        self.config.names = 'dummy'
        self.config.units = '-'
        
        self.states.names = ['dummy']
        self.states.units = ['-']

        self.params.names = ['n']
        self.params.units = ['-']
        self.params.min = [0.5]
        self.params.max = [5.]
        self.params.default = [2.3]

        self.params.reset()


    def run(self):
        P = self.inputs.data[:,0] 
        PE = self.inputs.data[:,1]
        n = self.params['n']
        Q = P*(1.-1./(1.+(P/PE)**n)**(1./n))
        E = P-Q
        self.outputs.data[:, 0] = Q

        if self.outputs.nvar > 1:
            self.outputs.data[:, 1] = E



class CalibrationTurcMezentsev(Calibration):

    def __init__(self, timeit=False):

        tm = TurcMezentsev()

        Calibration.__init__(self, 
            model = tm, \
            ncalparams = 1, \
            timeit = timeit)

        self.calparams_means.data =  [2.3]
        self.calparams_stdevs.data = [1.]


    def cal2true(self, calparams):
        return calparams


