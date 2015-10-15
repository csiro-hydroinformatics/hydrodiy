import math
import time
import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell as fmin

def atleast2d(data):
    data2d = np.ascontiguousarray(np.atleast_2d(data))

    shp = np.array(data).shape
    if len(shp) == 1:
        data2d = data2d.T
    elif len(shp) in [0, 2]:
        pass
    else:
        raise ValueError('len(data.shape)(%d) not in [0, 1, 2]' % \
                len(shp))

    return data2d


def errfun_abg(obs, sim, alpha, beta, gamma):

    E1 = 0.
    if alpha > 1e-10:
        E1 = np.sum(obs**gamma-sim**gamma)

    E2 = 0.
    if alpha < 1-1e-10:
        E2 = np.sum(np.sort(obs)**gamma-np.sort(sim)**gamma)

    mobs = np.mean(obs)
    msim = np.mean(sim)
    B = abs(mobs-msim)/mobs

    return (alpha*E1 + (1-alpha)*E2)*(1+B**beta)


class Model:

    def __init__(self, name, nuhmaxlength, \
            nstates, \
            ntrueparams, ncalparams, \
            outputs_names, \
            calparams_mins, calparams_maxs, \
            calparams_means, calparams_stdevs):

        self.name = name
        self.outputs_names = outputs_names
        self.runtime = np.nan

        self.nuhmaxlength = nuhmaxlength
        self.nuhlength = 0
        self.uh = np.zeros(nuhmaxlength).astype(np.float64)
        self.statesuh = np.zeros(nuhmaxlength).astype(np.float64)

        self.nstates = nstates
        self.states = np.zeros(nstates).astype(np.float64)

        self.ntrueparams = ntrueparams
        self.trueparams = np.ones(ntrueparams) * np.nan

        self.ncalparams = ncalparams
        self.calparams = np.ones(ncalparams) * np.nan

        if len(calparams_mins) != ncalparams:
            raise ValueError('len(calparams_mins)(%d) != ncalparams(%d)' % (
                len(calparams_mins), ncalparams))
        self.calparams_mins = calparams_mins

        if len(calparams_maxs) != ncalparams:
            raise ValueError('len(calparams_maxs)(%d) != ncalparams(%d)' % (
                len(calparams_maxs), ncalparams))
        self.calparams_maxs = calparams_maxs

        if len(calparams_means) != ncalparams:
            raise ValueError('len(calparams_means)(%d) != ncalparams(%d)' % (
                len(calparams_means), ncalparams))
        self.calparams_means = calparams_means

        if np.prod(np.array(calparams_stdevs).shape) != ncalparams**2:
            raise ValueError('np.prod(calparams_means.shape)(%d) != ncalparams**2(%d)' % (
                np.prod(np.array(calparams_means).shape), ncalparams**2))
        self.calparams_stdevs = calparams_stdevs


    def __str__(self):
        str = '\n%s model implementation\n' % self.name
        str += '  nuhlength  = %d\n' % self.nuhlength
        str += '  nuhmaxlength  = %d\n' % self.nuhmaxlength
        str += '  nstates = %d\n' % len(self.states)

        if hasattr(self, 'nout'):
            str += '  nout    = %d\n' % self.nout

        str += '  calparams  = ['
        for i in range(self.ncalparams):
            str += ' %0.3f' % self.calparams[i]
        str += ']\n'

        str += '  trueparams = ['
        for i in range(self.ntrueparams):
            str += ' %0.3f' % self.trueparams[i]
        str += ']\n'

        str += '  states     = ['
        for i in range(self.nstates):
            str += ' %0.3f' % self.states[i]
        str += ']\n'

        return str


    def create_outputs(self, nval, nout=1):
        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)


    def cal2true(self):
        trueparams = np.ones(len(self.ntrueparams)) * np.nan
        self.set_trueparams(trueparams)


    def set_uhparams(self):
        pass


    def set_trueparams(self, trueparams):
        trueparams = np.atleast_1d(trueparams).astype(np.float64)
        self.trueparams = np.atleast_1d(trueparams[:self.ntrueparams])
        self.set_uhparams()


    def set_calparams(self, calparams):
        calparams = np.atleast_1d(calparams)
        self.calparams = np.atleast_1d(calparams[:self.ncalparams])
        self.cal2true()


    def initialise(self, states=None, statesuh=None):
        if states is None:
            states = [0.] * self.nstates

        if statesuh is None:
            statesuh = [0.] * self.nuhmaxlength

        states = np.atleast_1d(states).astype(np.float64)
        self.states[:self.nstates] = states[:self.nstates]

        statesuh = np.array(statesuh).astype(np.float64)
        self.statesuh[:self.nuhlength] = statesuh[:self.nuhlength]


    def run(self, inputs):
        pass


    def get_outputs(self):
        outputs = pd.DataFrame(self.outputs)
        outputs.columns = self.outputs_names[:self.nout]

        return outputs


    def get_calparams_samples(self, nsamples):
        return np.random.multivariate_normal(self.calparams_means, \
                self.calparams_stdevs, \
                nsamples)


    def calibrate(self, inputs, observations, idx_cal,\
            ioutputs_cal=[0],\
            errfun=errfun_abg,
            errfun_args=(0.1, 2, 0.5,),\
            nsamples=500,\
            iprint=10, \
            minimize=True, \
            timeit=False):

        # check inputs
        observations = atleast2d(observations)
        inputs = atleast2d(inputs)

        nvalobs = observations.shape[0]
        nvalinputs = inputs.shape[0]
        nobs = observations.shape[1]

        if len(ioutputs_cal) != nobs:
            raise ValueError(('len(ioutputs_cal)(%d) != ' \
                'nobs(%d)') % (
                len(ioutputs_cal), nobs))

        if np.max(idx_cal) >= nvalobs:
            raise ValueError(('np.max(idx_cal)(%d) >= ' \
                    'nvalobs(%d)') % (
                np.max(idx_cal), nvalobs))

        if nvalobs != nvalinputs:
            raise ValueError('nvalobs(%d) != nvalinputs(%d)' % \
                    nvalobs, nvalinputs)

        # Allocate memory
        noutputs = np.max(ioutputs_cal) + 1
        self.create_outputs(nvalobs, noutputs)

        # Define objective function
        def objfun(calparams):

            if timeit:
                t0 = time.time()

            self.initialise()
            self.set_calparams(calparams)
            self.run(inputs)

            if timeit:
                t1 = time.time()
                self.runtime = 1000 * (t1-t0)

            if not errfun_args is None:
                ofun = errfun(observations[idx_cal, :], \
                        self.outputs[idx_cal, ioutputs_cal], \
                        *errfun_args)
            else:
                ofun = errfun(observations[idx_cal, :], \
                        self.outputs[idx_cal, \
                        ioutputs_cal])

            if not minimize:
                ofun *= -1

            return ofun


        # Systematic exploration
        calparams_explore = self.get_calparams_samples(nsamples)
        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        calparams_ini = []
        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = objfun(calparams)
            ofun_explore[i] = ofun

            if i%iprint == 0 and iprint>0:
                print('Exploration %d/%d : %03.3e [%s] ~ %0.2fms' % ( \
                        i, nsamples, ofun, calparams, self.runtime))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_ini = calparams

        # Minisation
        if iprint>0:
            print('\nOptimization start: %03.3e [%s] ~ %0.2fms\n' % ( \
                    ofun_min, calparams_ini, self.runtime))

        calparams_final = fmin(objfun, calparams_ini, disp=iprint>0)
        ofun_final = objfun(calparams_final)
        outputs_final = self.outputs

        if iprint>0:
            print('\nOptimization end: %03.3e [%s] ~ %0.2fms\n' % ( \
                    ofun_final, calparams_final, self.runtime))

        return calparams_final, outputs_final, ofun_final, \
                calparams_explore, ofun_explore

