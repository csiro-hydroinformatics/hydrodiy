import math
import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell as fmin


class Model:

    def __init__(self, name, nuhmax, nstates, \
            ntrueparams, ncalparams, \
            outputs_names, \
            calparams_mins, calparams_maxs, \
            calparams_means, calparams_stdevs):

        self.name = name
        self.outputs_names = outputs_names
        
        self.nuh = 0
        self.nuhmax = nuhmax
        self.uh = np.zeros(self.nuhmax).astype(np.float64)
        self.statesuh = np.zeros(nuhmax).astype(np.float64)

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
        str += '  nuhmax  = %d\n' % self.nuhmax
        str += '  nstates = %d\n' % len(self.states)
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
            statesuh = [0.] * self.nuh

        states = np.atleast_1d(states).astype(np.float64)
        self.states[:self.nstates] = states[:self.nstates]

        statesuh = np.array(statesuh).astype(np.float64)
        self.statesuh[:self.nuh] = statesuh[:self.nuh]


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


    def calib(self, inputs, observations, errfun, idx_cal,\
            ioutputs_cal=[0],\
            errfun_args=None,\
            nsamples=500,\
            iprint=10,
            minimize=True):

        # Check inputs
        if len(ioutputs_cal) != np.atleast_2d(observations).shape[1]:
            raise ValueError('len(ioutputs_cal)(%d) != observations.shape[1](%d)' % (
                len(ioutputs_cal), observations.shape[1]))

        if np.max(idx_cal) >= observations.shape[0]:
            raise ValueError('np.max(idx_cal)(%d) >= observations.shape[0](%d)' % (
                np.max(idx_cal), observations.shape[0]))

        # Allocate memory
        self.create_outputs(len(inputs), np.max(ioutputs_cal)+1)

        # Define objective function
        def objfun(calparams):
            self.initialise()
            self.set_calparams(calparams)
            self.run(inputs)

            if not errfun_args is None:
                ofun = errfun(observations[idx_cal, :], 
                        self.outputs[idx_cal, ioutputs_cal], *errfun_args) 
            else:
                ofun = errfun(observations[idx_cal, :], 
                        self.outputs[idx_cal, ioutputs_cal]) 

            if not minimize:
                ofun *= -1
            
            #print('Parameter optimization : %03.3e [%s]' % (ofun, calparams))

            return ofun


        # Prefiltering
        calparams_explore = self.get_calparams_samples(nsamples)
        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        calparams_ini = []
        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = objfun(calparams)
            ofun_explore[i] = ofun

            if i%iprint == 0 and iprint>0:
                print('Parameter exploration %d/%d : %03.3e [%s]' % (i, nsamples, ofun, calparams))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_ini = calparams 

        # Minisation
        if iprint>0:
            print('\nOptimization start: %03.3e [%s]\n' % (ofun_min, calparams_ini))
        
        calparams_final = fmin(objfun, calparams_ini, disp=iprint>0)
        ofun_final = objfun(calparams_final)
        outputs_final = self.outputs

        if iprint>0:
            print('\nOptimization end: %03.3e [%s]\n' % (ofun_final, calparams_final))

        return calparams_final, outputs_final, ofun_final, calparams_explore, ofun_explore

