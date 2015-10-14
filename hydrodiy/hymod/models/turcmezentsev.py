
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model

class TurcMezentsev(Model):

    def __init__(self):
        Model.__init__(self, 'turcmezentsev', 0, 0, \
            1, 1, \
            ['Q[mm/m]'],
            [-1], \
            [2], \
            [1], \
            [0.5])


    def run(self, inputs):
        P = inputs[:,0] 
        PE = inputs[:,1]
        exponent = self.trueparams[0]
        self.outputs = P*(1.-1./(1.+(P/PE)**exponent)**(1./exponent))


    def cal2true(self):
        x = self.calparams
        self.trueparams = np.array([np.exp(xt[0])])

