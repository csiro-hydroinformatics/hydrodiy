
import math
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model

class TurcMezentsev(Model):

    def __init__(self):
        Model.__init__(self, 'turcmezentsev', 1, ['dummy'], [0], 2, 0, 0, \
            1, 1, \
            ['n'], \
            ['Q[mm/m]'], \
            [0.833], \
            [[0.3]], \
            [0.5], \
            [4.], \
            [2.3])


    def run(self, inputs):

        if inputs.shape[1] != self.ninputs:
            raise ModelError(self.name, 
                    ierr_id='ESIZE_INPUTS', \
                    message='returned from TurcMezentsev.run')

        P = inputs[:,0] 
        PE = inputs[:,1]
        exponent = self.trueparams[0]
        self.outputs = P*(1.-1./(1.+(P/PE)**exponent)**(1./exponent))


    def cal2true(self, calparams):
        return math.exp(calparams[0])
        

