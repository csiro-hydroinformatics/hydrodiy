import math
import random
import time
import numpy as np
import pandas as pd

from scipy.optimize import fmin_powell as fmin

import c_hymod_models_utils


class ModelError(Exception):

    def __init__(self, model, \
            ierr=None, \
            ierr_id=None, \
            message=''):

        self.model = model
        self.message = message

        # Set error messages
        error = np.zeros(50).astype(np.int32)
        c_hymod_models_utils.geterror(error)
        error = { \
                -1 : 'UNKNOWN', \
                error[0] : 'ESIZE_INPUTS', \
                error[1] : 'ESIZE_OUTPUTS', \
                error[2] : 'ESIZE_PARAMS', \
                error[3] : 'ESIZE_STATES', \
                error[4] : 'ESIZE_STATESUH', \
                error[5] : 'ESIZE_CONFIG', \
                error[10] : 'EINVAL', \
                error[11] : 'EMODEL_RUN' \
        }


        # Initialise
        if not ierr is None:
            self.ierr = ierr
            if ierr in error:
                self.ierr_id = error[ierr]
            else:
                self.ierr_id = error[-1]

            return

        if not ierr_id is None:
            self.ierr_id = ierr_id
            for ierr in error:
                if error[ierr] == ierr_id:
                    self.ierr = ierr
                    return

            self.ierr = -1
            return

        raise ValueError('Either one of ierr or ierr_id should be' + \
            ' different from None')



    def __str__(self):
        txt = '{0} model : error {1} (TAG {2}) : {3}'.format( \
                self.model,
                self.ierr,
                self.ierr_id,
                self.message)

        return repr(txt)



def checklength(x, nx, model, ierr_id, message):
    if len(x) != nx:
        moderr = ModelError(model.name,
            ierr_id = ierr_id,
            message='{0}, len(x)({1}) != {2}'.format(message, len(x), nx))
        raise moderr


def vect2txt(x):
    txt = ''
    x = np.atleast_1d(x)
    for i in range(len(x)):
        txt += ' %0.2f' % x[i]
    return txt


class Model(object):

    def __init__(self, name, \
            nconfig, \
            config_names, \
            config_default, \
            ninputs, \
            nuhmaxlength, \
            nstates, \
            ntrueparams, \
            ncalparams, \
            trueparams_names, \
            outputs_names, \
            calparams_means, \
            calparams_stdevs, \
            trueparams_mins, \
            trueparams_maxs, \
            trueparams_default):

        self.name = name
        self.ninputs = ninputs
        self.outputs_names = outputs_names
        self.hitbounds = False

        # Config data
        self.nconfig = nconfig
        self.config_names = config_names
        checklength(self.config_names, nconfig, self, \
                'ESIZE_CONFIG', 'Problem with config_names')

        self.config_default = config_default
        checklength(self.config_default, nconfig, self, \
                'ESIZE_CONFIG', 'Problem with config_default')
        self.set_config_default()

        # UH data
        self.nuhmaxlength = nuhmaxlength
        self.nuhlength = 0
        self.uh = np.zeros(nuhmaxlength).astype(np.float64)
        self.statesuh = np.zeros(nuhmaxlength).astype(np.float64)

        # States data
        self.nstates = nstates
        self.states = np.zeros(nstates).astype(np.float64)

        # Parameter data
        self.ntrueparams = ntrueparams
        self.trueparams = np.ones(ntrueparams) * np.nan

        self.trueparams_names = trueparams_names
        checklength(self.trueparams_names, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_names')

        self.trueparams_default = np.atleast_1d(trueparams_default).astype(np.float64)
        checklength(self.trueparams_default, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_default')

        self.trueparams_mins = np.atleast_1d(trueparams_mins).astype(np.float64)
        checklength(self.trueparams_default, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_mins')

        self.trueparams_maxs = np.atleast_1d(trueparams_maxs).astype(np.float64)
        checklength(self.trueparams_maxs, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_maxs')

        self.ncalparams = ncalparams
        self.calparams = np.ones(ncalparams) * np.nan

        self.calparams_means = np.atleast_1d(calparams_means).astype(np.float64)
        checklength(self.calparams_means, ncalparams, self, \
                'ESIZE_PARAMS', 'Problem with calparams_means')

        self.calparams_stdevs = np.atleast_2d(calparams_stdevs)
        checklength(self.calparams_stdevs.flat[:], \
                ncalparams*ncalparams, self, \
                'ESIZE_PARAMS', 'Problem with calparams_stdevs')

        self.set_trueparams_default()


    def __str__(self):
        str = '\n{0} model implementation\n'.format(self.name)
        str += '  nconfig  = {0}\n'.format(self.nconfig)
        str += '  nuhlength  = {0}\n'.format(self.nuhlength)
        str += '  nuhmaxlength  = {0}\n'.format(self.nuhmaxlength)
        str += '  nstates = {0}\n'.format(len(self.states))

        if hasattr(self, 'nout'):
            str += '  nout    = {0}\n'.format(self.nout)

        if hasattr(self, 'config'):
            str += '  config  = ['
            for i in range(self.nconfig):
                str += ' {0}:{1:.3f}'.format(self.config_names[i], \
                    self.config[i])
            str += ']\n'

        str += '  calparams  = ['
        for i in range(self.ncalparams):
            str += ' {0:.3f}'.format(self.calparams[i])
        str += ']\n'

        str += '  trueparams = ['
        for i in range(self.ntrueparams):
            str += ' {0}:{1:.3f}'.format(self.trueparams_names[i], \
                self.trueparams[i])
        str += ']\n'

        str += '  states     = ['
        for i in range(self.nstates):
            str += ' {0:.3f}'.format(self.states[i])
        str += ']\n'

        return str


    def create_outputs(self, nval, nout=1):
        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)


    def cal2true(self, calparams):
        ''' No transform by default '''
        return calparams


    def set_uhparams(self):
        pass


    def set_config(self, config):
        self.config = np.atleast_1d(config[:self.nconfig]).astype(np.float64)


    def set_config_default(self):
        self.set_config(self.config_default)


    def set_trueparams_default(self):
        self.set_trueparams(self.trueparams_default)


    def set_trueparams(self, trueparams):
        trueparams = np.atleast_1d(trueparams).astype(np.float64)
        self.trueparams = np.atleast_1d(trueparams[:self.ntrueparams])

        # Check parameters bounds
        self.trueparams = np.clip(self.trueparams, \
            self.trueparams_mins, \
            self.trueparams_maxs)

        self.set_uhparams()


    def set_calparams(self, calparams):
        calparams = np.atleast_1d(calparams)
        self.calparams = np.atleast_1d(calparams[:self.ncalparams])

        trueparams = self.cal2true(self.calparams)
        self.set_trueparams(trueparams)


    def initialise(self, states=None, statesuh=None):
        if states is None:
            states = [0.] * self.nstates

        if statesuh is None:
            statesuh = [0.] * self.nuhmaxlength

        states = np.atleast_1d(states).astype(np.float64)
        states = np.atleast_1d(states[:self.nstates])
        self.states[:self.nstates] = states

        statesuh = np.array(statesuh).astype(np.float64)
        statesuh = np.atleast_1d(statesuh[:self.nuhlength])
        self.statesuh[:self.nuhlength] = statesuh


    def run(self, inputs):
        pass


    def get_outputs(self):
        outputs = pd.DataFrame(self.outputs)
        outputs.columns = self.outputs_names[:self.nout]

        return outputs


    def get_calparams_samples(self, nsamples, seed=333):

        # Save random state
        random_state = random.getstate()

        # Set seed
        np.random.seed(seed)

        # sample parameters
        samples = np.random.multivariate_normal(self.calparams_means, \
                self.calparams_stdevs, \
                nsamples)

        samples = np.atleast_2d(samples)

        if samples.shape[0] == 1:
            samples = samples.T

        # Reset random state
        random.setstate(random_state)

        return samples



class Calibration(object):

    def __init__(self, model, inputs, observations,
            errfun=None, \
            minimize=True, \
            timeit=False):

        self.model = model
        self.minimize = minimize
        self.timeit = timeit
        self.ieval = 0
        self.iprint = 0

        if len(inputs.shape) <= 1:
            self.inputs = np.atleast_2d(inputs).T
        else:
            self.inputs = inputs

        if len(observations.shape) <= 1:
            self.observations = np.atleast_2d(observations).T
        else:
            self.observations = observations

        self.nval = self.observations.shape[0]
        self.nobs = self.observations.shape[1]
        nvalinputs = self.inputs.shape[0]

        if self.nval != nvalinputs:
            raise ModelError(self.name,
                ierr_id='ESIZE_INPUTS',
                message = \
                'model.calibrate, nval({0}) != nvalinputs({1})'.format( \
                self.nval, nvalinputs))

        # Allocate memory
        self.model.create_outputs(self.nval, self.nobs)

        # Define objective function
        self.timeit = timeit
        self.runtime = np.nan

        def objfun(calparams):

            if self.timeit:
                t0 = time.time()

            self.model.set_calparams(calparams)
            self.model.initialise()
            self.model.run(inputs)
            self.ieval += 1

            if self.timeit:
                t1 = time.time()
                self.runtime = (t1-t0)*1000

            ofun = self.errfun(self.observations[self.idx_cal, :], \
                    self.model.outputs[self.idx_cal, :])

            if not self.minimize:
                ofun *= -1

            if self.iprint>0:
                if self.ieval%self.iprint == 0:
                    print('Fit %d : %03.3e [%s] ~ %0.2f ms' % ( \
                        self.ieval, ofun, vect2txt(calparams), \
                        self.runtime))


            return ofun

        self.objfun = objfun

        self.set_idx_cal()
        self.set_errfun()


    def __str__(self):
        str = 'Calibration instance for model {0}\n'.format(self.model.name)
        str += '    nval = {0}\n'.format(self.nval)
        str += '    nobs = {0}\n'.format(self.nobs)


    def reset_ieval(self):
        self.ieval = 0


    def set_idx_cal(self, idx_cal=None):
        if idx_cal is None:
            idx_cal = np.where(np.ones_like(self.observations) == 1)[0]

        if idx_cal.dtype == np.dtype('bool'):
            self.idx_cal = np.where(idx_cal)[0]
        else:
            self.idx_cal = idx_cal

        if np.max(self.idx_cal) >= self.nval:
            raise ModelError(self.name,
                ierr_id='ESIZE_INPUTS',
                message = \
                'model.calibrate, np.max(idx_cal)({0}) >= nval({1})'.format( \
                np.max(self.idx_cal), self.nval))


    def set_errfun(self, errfun=None):
        if errfun is None:
            def errfun(obs, sim):
                err = (obs-sim)
                return np.sum(err*err)
            self.errfun = errfun
        else:
            self.errfun = errfun


    def explore(self, calparams_explore=None, \
            nsamples = None, \
            iprint=0, \
            seed=333):

        self.iprint = iprint

        if nsamples is None:
            nsamples = int(200 * math.sqrt(self.model.ncalparams))

        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        if calparams_explore is None:
            calparams_explore = self.model.get_calparams_samples(nsamples, seed)
        else:
            calparams_explore = np.atleast_2d(calparams_explore)
            if calparams_explore.shape[0] == 1:
                calparams_explore = calparams_explore.T

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = self.objfun(calparams)
            ofun_explore[i] = ofun
            self.ieval += 1

            if self.iprint>0:
                if self.ieval%self.iprint == 0:
                    print('Exploration %d/%d : %03.3e [%s] ~ %0.2f ms' % ( \
                        self.ieval, nsamples, ofun, vect2txt(calparams), \
                        self.runtime))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        return calparams_best, calparams_explore, ofun_explore


    def fit(self, calparams_ini, iprint=0):

        self.iprint = iprint

        if self.iprint>0:
            ofun_ini = self.objfun(calparams_ini)
            print('\nOptimization start: %03.3e [%s] ~ %0.2f ms\n' % ( \
                    ofun_ini, vect2txt(calparams_ini), self.runtime))

        calparams_final = fmin(self.objfun, calparams_ini, disp=self.iprint>0)
        ofun_final = self.objfun(calparams_final)
        outputs_final = self.model.outputs

        if self.iprint>0:
            print('\nOptimization end: %03.3e [%s] ~ %0.2f ms\n' % ( \
                    ofun_final, vect2txt(calparams_final), self.runtime))

        return calparams_final, outputs_final, ofun_final

