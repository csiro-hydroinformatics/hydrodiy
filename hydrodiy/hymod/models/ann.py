
import numpy as np
import pandas as pd

from hystat import sutils

from hymod.model import Model
from hymod.calibration import Calibration


def standardize(X, cst=None):

    U = X
    if not cst is None:
        U = np.log(X+cst)

    U = np.atleast_2d(U)
    if U.shape[0] == 1:
        U = U.T

    mu = np.nanmean(U, 0)
    su = np.nanstd(U, 0)
    Un = (U-mu)/su

    return Un, mu, su


def destandardize(Un, mu, su, cst=None):
    U = mu + Un * su
    if not cst is None:
        X = np.exp(U) - cst

    return X


class ANN(Model):

    def __init__(self, ninputs, nneurons):

        self.nneurons = nneurons

        nparams = (ninputs + 2) * nneurons + 1

        noutputs_max = nneurons + 1

        Model.__init__(self, 'ann',
            nconfig=1, \
            ninputs=ninputs, \
            nparams=nparams, \
            nstates=1, \
            noutputs_max = noutputs_max,
            inputs_names = ['I{0}'.format(i) for i in range(ninputs)], \
            outputs_names = ['L2N1'] + \
                ['L1N{0}'.format(i) for i in range(1, nneurons+1)])

        self.config.names = ['dummy']
        self.config.units = ['-']

        self.states.names = ['dummy']
        self.states.units = ['-']

        self.params.units = ['-'] * nparams
        self.params.min = [-10.] * nparams
        self.params.max = [10.] * nparams
        self.params.default = [0.] * nparams

        self.params.reset()


    def params2matrix(self):
        nneurons = self._noutputs_max - 1
        ninputs = self.ninputs

        params = self.params.data

        n1 = ninputs*nneurons

        # Parameter for first layer
        L1M = params[:n1].reshape(ninputs, nneurons)
        L1C = params[n1:n1+nneurons].reshape(1, nneurons)

        # Parameter for second layer
        L2M = params[n1+nneurons:n1+2*nneurons].reshape(nneurons, 1)
        L2C = params[n1+2*nneurons:n1+2*nneurons+1].reshape(1, 1)

        return L1M, L1C, L2M, L2C


    def run(self):
        L1M, L1C, L2M, L2C = self.params2matrix()

        # First layer
        S = np.tanh(np.dot(self.inputs.data, L1M) + L1C)

        # Second layer
        O = np.dot(S, L2M) + L2C

        n3 = self.outputs.nvar
        self.outputs.data =  np.concatenate([O, S], axis=1)[:, :n3]



class CalibrationANN(Calibration):

    def __init__(self, ninputs, nneurons, timeit=False):

        ann = ANN(ninputs, nneurons)
        nparams = ann.params.nval

        Calibration.__init__(self,
            model = ann, \
            ncalparams = nparams, \
            timeit = timeit)

        self.calparams_means.data =  [0.] * nparams

        stdevs = np.eye(nparams).flat[:]
        self.calparams_stdevs.data = stdevs



