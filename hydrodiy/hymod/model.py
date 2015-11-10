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


class Vector(object):

    def __init__(self, id, nval):
        self._id = id
        self._nval = nval
        self._data = np.zeros(nval).astype(np.float64)
        self._names = ['X{0}'.format(i) for i in range(nval)]
        self._min = -np.inf * np.ones(nval).astype(np.float64)
        self._max = np.inf * np.ones(nval).astype(np.float64)
        self._default = np.nan * np.ones(nval).astype(np.float64)

    def __str__(self):
        str = '{0} : ['.format(self._id)
        str += ', '.join(['{0}:{1}'.format(self._names[i], self._data[i]) \
                for i in range(self._nval)])
        str += ']'

        return str

    def __set_value(self, target, source):
        _source = np.atleast_1d(source)

        if target != '_names':
            _source = _source.astype(np.float64)

        if len(_source) != self.nval:
            raise ValueError('Tried setting {0}, got wrong size ({1} instead of {2})'.format(\
                target, len(_source), self._nval))

        setattr(self, target, _source)


    def reset(self):
        self._data = self._default


    @property
    def nval(self):
        return self._nval


    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, value):

        if isinstance(value, dict):
            _value = np.zero(self._nval)
            for k in value:
                idx = np.where(self._names == k)[0]
                _value[idx] = value[k]
        else:
            _value = value

        self.__set_value('_data', _value)

        self._data = np.clip(self._data, self._min, self._max)
    

    @property
    def min(self):
        return self._min


    @min.setter
    def min(self, value):
        self.__set_value('_min', value)


    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self.__set_value('_max', value)

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self.__set_value('_default', value)
        self._default = np.clip(self._default, self._min, self._max)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self.__set_value('_names', value)


class Matrix(object):

    def __init__(self, id, nval, nvar):
        self._id = id
        self._nval = nval
        self._data = np.zeros((nval, nvar)).astype(np.float64)
        self._names = ['X{0}'.format(i) for i in range(nvar)]


    def __str__(self):
        str = '{0} : [nval={0} nvar={1}]'.format(self._id, self._nval, self.nvar)

        return str

    def reset(self):
        self._data = 0. * self._data

    @property
    def nval(self):
        return self._nval

    @property
    def nvar(self):
        return self._nvar

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)
        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[0] != self._nval
            raise ValueError(('Tried setting data,' + \
                    ' got wrong number of values ({1} instead of {2})'.format(\
                _value.shape[0], self._nval))

        if _value.shape[1] != self._nvar
            raise ValueError(('Tried setting data,' + \
                    ' got wrong number of variables ({1} instead of {2})'.format(\
                _value.shape[1], self._nvar))

        self._data = _value 



class Model(object):

    def __init__(self, name, \
            nconfig, \
            ninputs, \
            noutputs, \
            nuhmaxlength, \
            nstates, \
            nparams):

        self._name = name
        self._nval = nval
        self._ninputs = ninputs
        self._nuhmaxlength = nuhmaxlength
        self._nuhlength = 0

        self._config = Vector('config', nconfig)
        self._states = Vector('states', nstates)
        self._params = Vector('params', nparams)


    def __str__(self):
        str = '\n{0} model implementation\n'.format(self._name)
        str += '  ninputs      = {0}\n'.format(self._ninputs)
        str += '  nuhmaxlength = {0}\n'.format(self._nuhmaxlength)
        str += '  nuhlength    = {0}\n'.format(self._nuhlength)

        if hasattr(self, '_noutputs'):
            str += '  noutputs    = {0}\n'.format(self._noutputs)

        if hasattr(self, '_nval'):
            str += '  nval    = {0}\n'.format(self._nval)

        str += '  {0}\n'.format(self._config)
        str += '  {0}\n'.format(self._states)
        str += '  {0}\n'.format(self._params)

        return str


    @property
    def config(self):
        return self._config.data


    @config.setter
    def config(self, value):
        self._config.data = value


    @property
    def states(self):
        return self._states.data


    @states.setter
    def states(self, value):
        self._states.data = value


    @property
    def params(self):
        return self._params.data


    @params.setter
    def params(self, value):
        self._params.data = value
        self.set_uhparams()


    @property
    def inputs(self):
        return self._inputs.data


    @inputs.setter
    def inputs(self, value):
        self._inputs.data = value


    @property
    def outputs(self):
        return self._outputs.data


    @outputs.setter
    def outputs(self, value):
        self._outputs.data = value


    def allocate(self, nval, noutputs=1):
        self._noutputs = noutputs
        self._nval = nval
        self._inputs = Matrix('inputs', nval, self._ninputs)
        self._outputs = Matrix('outputs', nval, self._noutputs)


    def set_uhparams(self):
        pass


    def initialise(self, states=None, statesuh=None):
        if states is None:
            states = [0.] * self.nstates
        self.states = states

        if statesuh is None:
            statesuh = [0.] * self.nuhmaxlength
        self.statesuh = statesuh


    def run(self, inputs):
        pass



class Calibration(object):

    def __init__(self, model, \
            ncalparams, \
            observations, \
            errfun=None, \
            minimize=True, \
            timeit=False):

        self.model = model
        self.minimize = minimize
        self.timeit = timeit
        self.ieval = 0
        self.iprint = 0

        self.observations = Matrix('observations', , self.model._outputs.nvar)

        self.calparams = Vector('calparams', ncalparams)
        self.calparams_means = Vector('calparams_means', ncalparams)
        self.calparams_stdev = Vector('calparams_stdev', \ 
                ncalparams*ncalparams)
    
    @property
    def inputs(self):
        return self.model._inputs.data
 
    @inputs.setter
    def inputs(self, value):
        self.model._inputs.data = value

    @property
    def outputs(self):
        return self._outputs.data


       
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

    def calparams_samples(self, nsamples, seed=333):

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

