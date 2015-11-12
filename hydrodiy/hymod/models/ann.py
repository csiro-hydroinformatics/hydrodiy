
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration


class ANN(Model):

    def __init__(self, ninputs, nneurons):

        self.nneurons = nneurons

        nparams = (ninputs + 2) * nneurons + 1
        nconfig = ninputs * 2

        Model.__init__(self, 'ann',
            nconfig=nconfig, \
            ninputs=ninputs, \
            nparams=nparams, \
            nstates=nneurons, \
            noutputs_max = 1,
            inputs_names = ['I{0}'.format(i) for i in range(ninputs)], \
            outputs_names = ['O'])

        self.config.names = ['C{0}'.format(i) for i in range(nconfig)]
        self.config.units = ['-'] * nconfig

        self.states.names = ['S{0}'.format(i) for i in range(nneurons)]
        self.states.units = ['-'] * nneurons

        self.params.names = ['P{0}'.format(i) for i in range(nparams)]
        self.params.units = ['-'] * nparams
        self.params.min = [-10.] * nparams
        self.params.max = [10.] * nparams
        self.params.default = [0.] * nparams

        self.params.reset()


    def run(self):

        ninputs = self.ninputs
        nneurons = self.nneurons

        inputs = np.append(self.inputs.data, \
                np.ones((self.inputs.nval, 1)), 1)

        # First layer
        n1 = ninputs * nneurons + 1
        M = self.params.data[:n1].reshape((ninputs+1, nneurons))
        S = np.tanh(np.dot(inputs, M)



class CalibrationANN(Calibration):

    def __init__(self, ninputs, nneurons, timeit=False):

        ann = ANN(ninputs, nneurons)
        nparams = ann.params.nval

        Calibration.__init__(self,
            model = ann, \
            ncalparams = nparams, \
            timeit = timeit)

        self.calparams_means.data =  [0.] * nparams

        stdevs = [0.] * nparams * nparams
        self.calparams_stdevs.data = stdevs



