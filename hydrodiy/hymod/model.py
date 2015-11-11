import math
import random
import time
import numpy as np

from scipy.optimize import fmin_powell as fmin

import c_hymod_models_utils


class Vector(object):

    def __init__(self, id, nval):
        self._id = id
        self._nval = nval
        self._data = np.zeros(nval).astype(np.float64)
        self._names = ['X{0}'.format(i) for i in range(nval)]
        self._units = ['-'] * nval
        self._min = -np.inf * np.ones(nval).astype(np.float64)
        self._max = np.inf * np.ones(nval).astype(np.float64)
        self._default = np.nan * np.ones(nval).astype(np.float64)

    def __str__(self):
        str = '{0} : {{'.format(self._id)
        str += ', '.join(['{0}: {1}[{2}]'.format( \
                self._names[i], self._data[i], self._units[i]) \
                for i in range(self._nval)])
        str += '}'

        return str

    def __getitem__(self, name):
        if name in self._names:
            kx = np.where(name == self._names)[0]
            return self._data[kx]
        else:
            raise ValueError('Name {0} not in the vector names'.format(name))


    def __setitem__(self, name, value):
        if name in self._names:
            kx = np.where(name == self._names)[0]
            self._data[kx] = value
        else:
            raise ValueError('Name {0} not in the vector names'.format(name))


    def __set_value(self, target, source):
        _source = np.atleast_1d(source)

        if not target in ['_names', '_units']:
            _source = _source.astype(np.float64)

        if len(_source) != self.nval:
            raise ValueError(('Tried setting {0}, ' + \
                'got wrong size ({1} instead of {2})').format(\
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
        self.__set_value('_data', value)
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


    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self.__set_value('_units', value)



class Matrix(object):

    def __init__(self, id, nval, nvar, data=None):
        self._id = id

        if nval is not None and nvar is not None:
            self._nval = nval
            self._nvar = nvar
            self._data = np.zeros((nval, nvar)).astype(np.float64)

        elif data is not None:
            _data = np.atleast_2d(data)
            if _data.shape[0] == 1:
                _data = _data.T

            self._data = _data
            self._nval = _data.shape[0]
            self._nvar = _data.shape[1]

        else:
            raise ValueError('Wrong arguments to Matrix.__init__')

        self._names = ['X{0}'.format(i) for i in range(self._nvar)]


    @classmethod
    def fromdims(cls, id, nval, nvar):
        return cls(id, nval, nvar, None)


    @classmethod
    def fromdata(cls, id, data):
        return cls(id, None, None, data)


    def __str__(self):
        str = '{0} : nval={1} nvar={2} ['.format( \
            self._id, self._nval, self._nvar)
        str += ' '.join(self._names)
        str += ']'

        return str

    def __getitem__(self, name):
        if name in self._names:
            kx = np.where(name == self._names)[0][0]
            return self._data[:, kx]
        else:
            raise ValueError('Name {0} not in the matrix names'.format(name))


    def __setitem__(self, name, value):
        if name in self._names:
            kx = np.where(name == self._names)[0][0]
            self._data[:, kx] = value
        else:
            raise ValueError('Name {0} not in the vector names'.format(name))


    def reset(self):
        self._data = 0. * self._data


    @property
    def nval(self):
        return self._nval


    @property
    def nvar(self):
        return self._nvar


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = np.atleast_1d(value)

        if len(self._names) != self._nvar:
            raise ValueError(('Tried setting _names, ' + \
                'got wrong size ({0} instead of {1})').format(\
                len(self._names), self._nvar))


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)

        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[0] != self._nval:
            raise ValueError(('Tried setting _data,' + \
                    ' got wrong number of values ' + \
                    '({0} instead of {1})').format( \
                    _value.shape[0], self._nval))

        if _value.shape[1] != self._nvar:
            raise ValueError(('Tried setting _data,' + \
                    ' got wrong number of variables ' + \
                    '({0} instead of {1})').format( \
                    _value.shape[1], self._nvar))

        self._data = _value



class Model(object):

    def __init__(self, name, \
            nconfig, \
            ninputs, \
            nparams, \
            nstates, \
            noutputs_max):

        self._name = name
        self._ninputs = ninputs
        self._nuhlength = 0
        self._noutputs_max = noutputs_max

        self._config = Vector('config', nconfig)
        self._states = Vector('states', nstates)
        self._params = Vector('params', nparams)
        self._params_default = Vector('params_default', nparams)

        nuhmaxlength = c_hymod_models_utils.uh_getnuhmaxlength()
        self._statesuh = Vector('statesuh', nuhmaxlength)

        self._inputs = None
        self._outputs = None


    def __str__(self):
        str = '\n{0} model implementation\n'.format(self._name)
        str += '  nconfig      = {0}\n'.format(self._config.nval)
        str += '  ninputs      = {0}\n'.format(self._ninputs)
        str += '  nuhmaxlength = {0}\n'.format(self._statesuh.nval)
        str += '  nuhlength    = {0}\n'.format(self._nuhlength)
        str += '  nparams      = {0}\n'.format(self._params.nval)
        str += '  nstates      = {0}\n'.format(self._states.nval)

        if not self._outputs is None:
            str += '  noutputs     = {0}\n'.format(self._outputs.nvar)
            str += '  nval         = {0}\n'.format(self._outputs.nval)

        str += '  {0}\n'.format(self._config)
        str += '  {0}\n'.format(self._states)
        str += '  {0}\n'.format(self._params)

        return str


    @property
    def name(self):
        return self._name


    @property
    def nuhlength(self):
        return self._nuhlength


    @property
    def ninputs(self):
        return self._ninputs


    @property
    def config(self):
        return self._config


    @property
    def states(self):
        return self._states


    @property
    def params(self):
        return self._params


    @property
    def inputs(self):
        return self._inputs


    @property
    def outputs(self):
        return self._outputs


    def allocate(self, nval, noutputs=1):
        if noutputs > self._noutputs_max:
            raise ValueError('noutputs({0}) > noutputs_max({1})'.format( \
                noutputs, self._noutputs_max))

        self._noutputs = noutputs
        self._inputs = Matrix.fromdims('inputs', nval, self._ninputs)
        self._outputs = Matrix.fromdims('outputs', nval, self._noutputs)


    def set_uhparams(self):
        pass


    def initialise(self, states=None, statesuh=None):
        if states is None:
            states = [0.] * self._states.nval
        self._states.data = states

        if statesuh is None:
            statesuh = [0.] * self._statesuh.nval
        self._statesuh.data = statesuh


    def run(self):
        pass



class Calibration(object):

    def __init__(self, model, \
            ncalparams, \
            observations, \
            errfun=None, \
            minimize=True, \
            optimizer=fmin, \
            initialise_model=True, \
            timeit=False):

        self._model = model
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._optimizer = optimizer
        self._initialise_model = initialise_model

        self._observations = Matrix.fromdata('observations', observations)

        self._calparams = Vector('calparams', ncalparams)
        self._calparams_means = Vector('calparams_means', ncalparams)
        self._calparams_stdevs = Vector('calparams_stdevs', \
                ncalparams*ncalparams)

        self.idx_cal = np.where(np.ones((self._observations.nval)) == 1)[0]

        def errfun(obs, sim, errparams):
            err = obs-sim
            return np.mean(err*err)
        self.errfun = errfun


    def __str__(self):
        str = 'Calibration instance for model {0}\n'.format(self._model.name)
        str += '  ncalparams : {0}\n'.format(self.calparams_means.nval)
        str += '  ieval      : {0}\n'.format(self._ieval)
        str += '  runtime    : {0}\n'.format(self._runtime)
        str += '  {0}\n'.format(self._model.params)
        str += '  {0}\n'.format(self.calparams_means)

        return str


    @property
    def calparams_means(self):
        return self._calparams_means


    @property
    def calparams_stdevs(self):
        return self._calparams_stdevs


    @property
    def model(self):
        return self._model


    @property
    def errfun(self):
        return self._errfun

    @errfun.setter
    def errfun(self, value):
        self._errfun = value

        def objfun(calparams):

            if self._timeit:
                t0 = time.time()

            params = self.cal2true(calparams)
            self._model._params.data = params

            if self._initialise_model:
                self._model.initialise()

            self._model.run()
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            errparams = self.cal2err(calparams)

            ofun = self._errfun(self._observations.data[self._idx_cal, :], \
                    self.model.outputs.data[self._idx_cal, :], errparams)

            if not self._minimize:
                ofun *= -1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('Fit {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams, \
                        self._runtime))

            return ofun

        self._objfun = objfun


    @property
    def idx_cal(self):
        return self._idx_cal

    @idx_cal.setter
    def idx_cal(self, value):

        if value.dtype == np.dtype('bool'):
            _idx_cal = np.where(value)[0]
        else:
            _idx_cal = value

        if np.max(_idx_cal) >= self._observations.nval:
            raise ValueError('Wrong values in idx_cal')

        self._idx_cal = _idx_cal


    def checkmodel(self):
        if self._model.inputs is None:
            raise ValueError(('No inputs data set for model {0}.' + \
                ' Please allocate').format(self._model.name))

        if self._model.outputs is None:
            raise ValueError(('No outputs data set for model {0}.' + \
                ' Please allocate').format(self._model.name))

        n1 = self._model.inputs.nval
        n2 = self._observations.nval
        if n1 != n2:
            raise ValueError(('model.inputs.nval({0}) !=' + \
                ' observations.nval({1})').format(n1, n2))


    def cal2true(self, calparams):
        return calparams


    def cal2err(self, calparams):
        return None


    def sample(self, nsamples, seed=333):

        # Save random state
        random_state = random.getstate()

        # Set seed
        np.random.seed(seed)

        ncalparams = self.calparams_means.nval

        # sample parameters
        samples = np.random.multivariate_normal(\
                self.calparams_means.data, \
                self.calparams_stdevs.data.reshape( \
                    ncalparams, ncalparams), \
                nsamples)

        samples = np.atleast_2d(samples)
        if samples.shape[0] == 1:
            samples = samples.T

        # Reset random state
        random.setstate(random_state)

        return samples


    def reset_ieval(self):
        self._ieval = 0


    def explore(self, \
            calparams_explore=None, \
            nsamples = None, \
            iprint=0, \
            seed=333):

        self.checkmodel()
        self._iprint = iprint

        if nsamples is None:
            ncalparams = self._calparams_means.nval
            nsamples = int(200 * math.sqrt(ncalparams))

        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        if calparams_explore is None:
            calparams_explore = self.sample(nsamples, seed)
        else:
            calparams_explore = np.atleast_2d(calparams_explore)
            if calparams_explore.shape[0] == 1:
                calparams_explore = calparams_explore.T

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = self._objfun(calparams)
            ofun_explore[i] = ofun
            self._ieval += 1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('Exploration {0}/{1} : {2:3.3e} {3} ~ {4:.2f} ms'.format( \
                        self._ieval, nsamples, ofun, self._calparams, \
                        self._runtime))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        return calparams_best, calparams_explore, ofun_explore


    def fit(self, calparams_ini, iprint=0, *args, **kwargs):

        self.checkmodel()
        self._iprint = iprint

        if self._iprint>0:
            ofun_ini = self._objfun(calparams_ini)

            self._calparams.data = calparams_ini
            print('\nFit start: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_ini, self._calparams, self._runtime))

        calparams_final = self._optimizer(self._objfun, \
                    calparams_ini, \
                    disp=self._iprint>0, *args, **kwargs)

        ofun_final = self._objfun(calparams_final)
        outputs_final = self.model.outputs.data

        if self._iprint>0:
            self._calparams.data = calparams_final
            print('\nFit final: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_final, self._calparams, self._runtime))

        return calparams_final, outputs_final, ofun_final

