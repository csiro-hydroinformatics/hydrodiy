''' Module providing data containers '''

import numpy as np

EPS = 1e-10


class Vector(object):
    ''' Vector data container. Implements min, max and default values. '''

    def __init__(self, names, defaults=None, mins=None, maxs=None, \
            check_hitbounds=False):

        # Set parameter names
        if names is None:
            names = []

        self._names = np.atleast_1d(names).flatten().astype(str).copy()
        nval = self._names.shape[0]

        # Record bounds hitting or not (useful for optimizers)
        self._check_hitbounds = bool(check_hitbounds)
        self._hitbounds = False

        # Set number of parameters
        if len(np.unique(self._names)) != nval:
            raise ValueError('Names are not unique: {0}'.format(names))
        self._nval = nval

        # Define names indexes
        self._names_index = {nm:i for nm, i \
                        in zip(self._names,  np.arange(nval))}

        # Set mins and maxs
        self._mins = -np.inf * np.ones(nval)
        self._maxs = np.inf * np.ones(nval)

        if not mins is None:
            # Just convert mins to proper array
            self._mins, _ = self.__checkvalues__(mins, False)

        if not maxs is None:
            # Convert maxs to proper array and checks it is greater than mins
            maxs2, hit = self.__checkvalues__(maxs, True)

            if hit:
                raise ValueError(('Expected maxs within [{0}, {1}],' +\
                    ' got {2}').format(self._mins, self._maxs, maxs))
            self._maxs = maxs2

        # Set defaults values
        if not defaults is None:
            self._defaults, hit = self.__checkvalues__(defaults, True)

            if hit:
                raise ValueError(('Expected defaults within [{0}, {1}],' +\
                    ' got {2}').format(self._mins, self._maxs, defaults))
        else:
            self._defaults = np.clip(np.zeros(nval), self._mins, self._maxs)

        # Set values to defaults
        self._values = self._defaults.copy()


    @classmethod
    def from_dict(cls, dct):
        ''' Create vector from  dictionary '''

        nval = dct['nval']
        names = []
        defaults = []
        mins = []
        maxs = []
        values = []
        for i in range(nval):
            names.append(dct['data'][i]['name'])
            defaults.append(dct['data'][i]['default'])
            mins.append(dct['data'][i]['min'])
            maxs.append(dct['data'][i]['max'])
            values.append(dct['data'][i]['value'])

        vect = Vector(names, defaults, mins, maxs, \
                            bool(dct['check_hitbounds']))
        vect._hitbounds = bool(dct['hitbounds'])
        vect.values = values

        return vect


    def __getattribute__(self, name):
        # Except _names and _hitbounds to avoid infinite recursion
        if name in ['_names', '_check_hitbounds', '_hitbounds']:
            return super(Vector, self).__getattribute__(name)

        if name in self._names:
            idx = self._names_index[name]
            return self._values[idx]

        return super(Vector, self).__getattribute__(name)


    def __setattr__(self, name, value):
        # Except _names and _hitbounds to avoid infinite recursion
        if name in ['_names', '_check_hitbounds', '_hitbounds']:
            super(Vector, self).__setattr__(name, value)
            return

        if name in self._names:
            value = np.float64(value)
            if np.isnan(value):
                raise ValueError('Cannot set value to nan')

            idx = self._names_index[name]

            # Check bounds if needed
            if self.check_hitbounds:
                self._hitbounds = (value < self._mins[idx]) \
                                            or (value > self._maxs[idx])

            # Store clipped value
            self.values[idx] = min(max(value, self.mins[idx]), self.maxs[idx])

        else:
            super(Vector, self).__setattr__(name, value)


    def __str__(self):
        return 'vector ['+', '.join(['{0}:{1:3.3e}'.format(key, self[key]) \
                                    for key in self.names]) + ']'


    def __checkvalues__(self, val, check_hitbounds):
        ''' Check vector is of proper length and contains no nan'''

        val = np.atleast_1d(val).flatten().astype(np.float64)

        if val.shape[0] != self.nval:
            raise ValueError('Expected vector of length {0}, got {1}'.format(\
                    self.nval, val.shape[0]))

        if np.any(np.isnan(val)):
            raise ValueError('Cannot process values with NaN')

        # Record hit bound if needed
        hit = False
        if check_hitbounds:
            hit = np.any((val < self._mins-EPS) | (val > self._maxs+EPS))

        # returns clipped values
        val = np.clip(val, self._mins, self._maxs)

        return val, hit


    def __setitem__(self, key, value):
        if not key in self._names_index:
            raise ValueError('Expected key in {0}, got {1}'.format(\
                self._names, key))

        setattr(self, key, value)


    def __getitem__(self, key):
        if not key in self._names_index:
            raise ValueError('Expected key in {0}, got {1}'.format(\
                self._names, key))

        return getattr(self, key)


    @property
    def nval(self):
        return self._nval


    @property
    def check_hitbounds(self):
        return self._check_hitbounds


    @property
    def hitbounds(self):
        return self._hitbounds


    @property
    def names(self):
        return self._names


    @property
    def mins(self):
        return self._mins


    @property
    def maxs(self):
        return self._maxs


    @property
    def defaults(self):
        return self._defaults


    @property
    def values(self):
        ''' Get data '''
        return self._values


    @values.setter
    def values(self, val):
        ''' Set data '''
        self._values, self._hitbounds = self.__checkvalues__(val, \
                                            self._check_hitbounds)


    def reset(self):
        ''' Reset vector values to defaults '''
        self.values = self.defaults


    def clone(self):
        ''' Clone the vector '''
        clone = Vector(self.names, self.defaults, self.mins, \
                    self.maxs, self.check_hitbounds)

        clone.values = self.values.copy()

        return clone


    def to_dict(self):
        ''' Write vector data to json format '''

        dct = {'nval': self.nval, 'hitbounds':self.hitbounds, \
                    'check_hitbounds':self._check_hitbounds, \
                    'data':[]}

        for i in range(self.nval):
            elem = {'name':self.names[i], \
                'value':self.values[i], \
                'min':self.mins[i], \
                'max':self.maxs[i], \
                'default':self.defaults[i]}
            dct['data'].append(elem)

        return dct


