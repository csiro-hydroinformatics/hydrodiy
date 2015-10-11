import math
import numpy as np
import pandas as pd


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
        self.statesuhini = np.zeros(nuhmax).astype(np.float64)

        self.nstates = nstates
        self.statesini = np.zeros(nstates).astype(np.float64)

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
        self.trueparams = np.atleast_1d(trueparams[:self.ntrueparams]).astype(np.float64)
        self.set_uhparams()


    def set_calparams(self, calparams):
        self.calparams = np.atleast_1d(calparams[:self.ncalparams])
        self.cal2true()


    def set_states(self, statesini=None, statesuhini=None):
        if statesini is None:
            statesini = [0.] * self.nstates

        if statesuhini is None:
            statesuhini = [0.] * self.nuh

        statesini = np.atleast_1d(statesini).astype(np.float64)
        self.statesini[:self.nstates] = statesini[:self.nstates]

        statesuhini = np.array(statesuhini).astype(np.float64)
        self.statesuhini[:self.nuh] = statesuhini[:self.nuh]


    def run(self, inputs):
        pass
   

    def get_outputs(self):
        outputs = pd.DataFrame(self.outputs)
        outputs.columns = self.outputs_names[:self.nout]

        return outputs


    def get_calparams_sample(self, nsamples):
        return np.random.multivariate_normal(self.calparams_means, \
                self.calparams_stdevs, \
                nsamples)


    def calib(self, inputs, outputs, objfun, objfun_args=None, nsamples=500):

        # Prefiltering
        samples = self.get_calparams_sample(nsamples)
        objfun_min = np.inf

        params0 = []
        for i in range(nsamples):
            params = samples[i,:]
            o = objfun(params, inputs, outputs, args)

            if o < objfun_min:
                objfun_min = o
                params0 = params 

        # Minisation
        params = fmin(objfun, params0, args)

        return params

