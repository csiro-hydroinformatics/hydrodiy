import math
import numpy as np

import c_hymod_models_utils

NUHMAXLENGTH = c_hymod_models_utils.uh_getnuhmaxlength()

class Vector(object):

    def __init__(self, id, nval):
        self._id = id
        self._nval = nval
        self._data = np.nan * np.ones(nval).astype(np.float64)
        self._names = ['X{0}'.format(i) for i in range(nval)]
        self._units = ['-'] * nval
        self._min = -np.inf * np.ones(nval).astype(np.float64)
        self._max = np.inf * np.ones(nval).astype(np.float64)
        self._default = np.nan * np.ones(nval).astype(np.float64)
        self._hitbounds = np.zeros(nval).astype(np.float64)
        self._model2vector = None


    def __str__(self):
        str = '{0} : {{'.format(self._id)
        str += ', '.join(['{0}: {1:3.3e}[{2}]'.format( \
                self._names[i], self._data[i], self._units[i]) \
                for i in range(self._nval)])
        str += '}'

        return str


    def __getitem__(self, name):
        if name in self._names:
            kx = np.where(name == self._names)[0]
            return self._data[kx]
        else:
            raise ValueError(('Vector {0}: name {1} not in' + \
                    ' the vector names').format(self._id, name))


    def __setitem__(self, name, value):
        if name in self._names:
            kx = np.where(name == self._names)[0]
            self._data[kx] = value
        else:
            raise ValueError(('Vector {0}: name {1} not in' + \
                    ' the vector names').format(self._id, name))


    def __set_data(self, target, source):
        _source = np.atleast_1d(source)

        if not target in ['_names', '_units']:
            _source = _source.astype(np.float64)

        if len(_source) != self.nval:
            raise ValueError(('Vector {0}: tried setting {1}, ' + \
                'got wrong size ({2} instead of {3})').format(\
                self._id, target, len(_source), self._nval))

        setattr(self, target, _source)


    def reset(self):
        self.data = self._default.copy()


    @property
    def nval(self):
        return self._nval


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.__set_data('_data', value)
        self._data = np.clip(self._data, self._min, self._max)

        idx = np.abs(self._data-self._min) < 1e-10
        idx = idx | (np.abs(self._data-self._max) < 1e-10)
        self._hitbounds[idx] = 1
        self._hitbounds[~idx] = 0

        if not self._model2vector is None:
            self._model2vector.post_setter()


    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self.__set_data('_min', value)


    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self.__set_data('_max', value)


    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self.__set_data('_default', value)
        self._default = np.clip(self._default, self._min, self._max)


    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self.__set_data('_names', value)


    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self.__set_data('_units', value)


    @property
    def hitbounds(self):
        return self._hitbounds


class Matrix(object):

    def __init__(self, id, nval, nvar, data=None):
        self._id = id

        if nval is not None and nvar is not None:
            self._nval = nval
            self._nvar = nvar
            self._data = np.nan * np.ones((nval, nvar)).astype(np.float64)

        elif data is not None:
            _data = np.atleast_2d(data)
            if _data.shape[0] == 1:
                _data = _data.T

            self._data = _data
            self._nval = _data.shape[0]
            self._nvar = _data.shape[1]

        else:
            raise ValueError(('{0} matrix, ' + \
                    'Wrong arguments to Matrix.__init__').format(self._id))

        self._names = ['V{0}'.format(i) for i in range(self._nvar)]


    @classmethod
    def fromdims(cls, id, nval, nvar):
        return cls(id, nval, nvar, None)


    @classmethod
    def fromdata(cls, id, data):
        return cls(id, None, None, data)


    def __str__(self):
        str = '{0} : nval={1} nvar={2} {{'.format( \
            self._id, self._nval, self._nvar)
        str += ', '.join(self._names)
        str += '}'

        return str


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
            raise ValueError(('{0} matrix: tried setting _names, ' + \
                'got wrong size ({1} instead of {2})').format(\
                self._id, len(self._names), self._nvar))


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        _value = np.atleast_2d(value)

        if _value.shape[0] == 1:
            _value = _value.T

        if _value.shape[0] != self._nval:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of values ' + \
                    '({1} instead of {2})').format( \
                    self._id, _value.shape[0], self._nval))

        if _value.shape[1] != self._nvar:
            raise ValueError(('{0} matrix: tried setting _data,' + \
                    ' got wrong number of variables ' + \
                    '({1} instead of {2})').format( \
                    self._id, _value.shape[1], self._nvar))

        self._data = _value



class Model2Vector(object):
    
    def __init__(self, model, vector):
        self.model = model
        self.vector = vector

    def post_setter(self):
        self.model.set_uh()


class Model(object):

    def __init__(self, name, \
            nconfig, \
            ninputs, \
            nparams, \
            nstates, \
            noutputs_max, \
            inputs_names, \
            outputs_names):

        self._name = name
        self._ninputs = ninputs
        self._nuhlength = 0
        self._noutputs_max = noutputs_max

        self._config = Vector('config', nconfig)
        self._states = Vector('states', nstates)
        self._params = Vector('params', nparams)
        self._params_default = Vector('params_default', nparams)

        model2vector = Model2Vector(self, self._params)
        self._params._model2vector = model2vector

        self._statesuh = Vector('statesuh', NUHMAXLENGTH)
        self._uh = Vector('uh', NUHMAXLENGTH)

        self._inputs = None
        self._outputs = None

        if len(inputs_names) != ninputs:
            raise ValueError(('Model {0}: len(inputs_names)({1}) != ' + \
                    'ninputs ({2})').format(self._name, \
                        len(inputs_names), ninputs))

        self._inputs_names = inputs_names


        if len(outputs_names) != noutputs_max:
            raise ValueError(('Model {0}: len(outputs_names)({1}) != ' + \
                    'noutputs_max ({2})').format(self._name, \
                        len(outputs_names), \
                        noutputs_max))

        self._outputs_names = outputs_names


    def __str__(self):
        str = '\n{0} model implementation\n'.format( \
            self._name)
        str += '  ninputs      = {0}\n'.format(self._ninputs)
        str += '  nuhmaxlength = {0}\n'.format(self._statesuh.nval)
        str += '  nuhlength    = {0}\n'.format(self._nuhlength)

        str += '  {0}\n'.format(self._config)
        str += '  {0}\n'.format(self._states)
        str += '  {0}\n'.format(self._params)

        if not self._inputs is None:
            str += '  {0}\n'.format(self._inputs)

        if not self._outputs is None:
            str += '  {0}\n'.format(self._outputs)

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
    def uh(self):
        return self._uh


    @property
    def statesuh(self):
        return self._statesuh


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
        if noutputs <= 0:
            raise ValueError('Number of outputs defined for model {0} should be >0')

        if noutputs > self._noutputs_max:
            raise ValueError(('Too many outputs defined for model {0}:' + \
                ' noutputs({1}) > noutputs_max({2})').format( \
                self._name, noutputs, self._noutputs_max))

        self._inputs = Matrix.fromdims('inputs', nval, self._ninputs)
        self._inputs.names = self._inputs_names

        self._outputs = Matrix.fromdims('outputs', nval, noutputs)
        self._outputs.names = self._outputs_names[:noutputs]


    def set_uh(self):
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


    def clone(self):

        model = Model(self._name, \
            self._config.nval, \
            self._ninputs, \
            self._params.nval, \
            self._states.nval, \
            self._noutputs_max, \
            self._inputs_names, \
            self._outputs_names)

        model._params.data = self._params.data.copy()
        model._states.data = self._states.data.copy()
        model._statesuh.data = self._statesuh.data.copy()
        model._config.data = self._config.data.copy()

        if not self._inputs is None:
            model.allocate(self._inputs.nval, self._outputs.nvar)
            model._inputs.data = self._inputs.data.copy()
            model._outputs.data = self._outputs.data.copy()

        return model


