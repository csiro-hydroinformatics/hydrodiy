''' Module providing data containers '''

import numpy as np

EPS = 1e-10


class Vector(object):
    ''' Vector data container. Implements min, max and default values. '''

    def __init__(self, names, defaults=None, mins=None, maxs=None, \
            hitbounds=False,
            post_setter_args=None):

        # Set parameter names
        if names is None:
            names = []

        self._names = np.atleast_1d(names).flatten().astype(str).copy()

        # Set number of parameters
        nval = self._names.shape[0]
        if len(np.unique(self._names)) != nval:
            raise ValueError('Names are not unique: {0}'.format(names))
        self._nval = nval

        self._hitbounds = hitbounds

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

        # Set post_setter arguments
        self._post_setter_args = post_setter_args


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

        vect = Vector(names, defaults, mins, maxs)
        vect.values = values
        vect._hitbounds = bool(dct['hitbounds'])

        return vect


    def __str__(self):
        return 'vector ['+', '.join(['{0}:{1:3.3e}'.format(key, self[key]) \
                                    for key in self.names]) + ']'


    def __findname__(self, key):
        idx = np.where(self.names == key)[0]
        if idx.shape[0] == 0:
            raise ValueError(('Expected key {0} in the' + \
                ' list of names {1}').format(key, self.names))

        return idx[0]


    def __checkvalues__(self, val, hitbounds=False):
        ''' Check vector is of proper length and contains no nan'''

        val = np.atleast_1d(val).flatten().astype(np.float64)

        if val.shape[0] != self.nval:
            raise ValueError('Expected vector of length {0}, got {1}'.format(\
                    self.nval, val.shape[0]))

        if np.any(np.isnan(val)):
            raise ValueError('Cannot process values with NaN')

        hit = False
        if hitbounds:
            hit = np.any((val < self._mins-EPS) | (val > self._maxs+EPS))

        val = np.clip(val, self._mins, self._maxs)

        return val, hit


    def __setitem__(self, key, value):
        idx = self.__findname__(key)
        value = np.float64(value)
        if np.isnan(value):
            raise ValueError('Cannot set value to nan')

        self._hitbounds = np.any((value < self._mins[idx]) | (value > self._maxs[idx]))
        self.values[idx] = np.clip(value, self.mins[idx], self.maxs[idx])


    def __getitem__(self, key):
        idx = self.__findname__(key)
        return self._values[idx]


    @property
    def nval(self):
        return self._nval


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
        self._values, self._hitbounds = self.__checkvalues__(val, True)

        # Function post setter
        if not self._post_setter_args is None:
            self.post_setter(*self._post_setter_args)


    def post_setter(self, *args):
        ''' Function run after setting parameter values '''
        pass


    def reset(self):
        ''' Reset vector values to defaults '''
        self.values = self.defaults


    def clone(self):
        ''' Clone the vector '''
        clone = Vector(self.names, self.defaults, self.mins, \
                    self.maxs)

        clone.values = self.values

        return clone


    def to_dict(self):
        ''' Write vector data to json format '''

        dct = {'nval': self.nval, 'hitbounds':self.hitbounds, \
                    'data':[]}

        for i in range(self.nval):
            elem = {'name':self.names[i], \
                'value':self.values[i], \
                'min':self.mins[i], \
                'max':self.maxs[i], \
                'default':self.defaults[i]}
            dct['data'].append(elem)

        return dct


