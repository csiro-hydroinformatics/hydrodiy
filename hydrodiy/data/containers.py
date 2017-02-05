import copy
import re
from itertools import product

import math
import numpy as np



class Vector(object):

    def __init__(self, names, defaults=None, mins=None, maxs=None, \
            hitbounds=False):

        # Set parameter names
        self._names = np.atleast_1d(names).flatten().astype(str)
        nval = self._names.shape[0]
        self._nval = nval

        self._hitbounds = hitbounds

        # Set mins
        if not mins is None:
            self._mins, _ = self.__checkvalues__(mins, False, False)
        else:
            self._mins = -np.inf * np.ones(nval)

        # Set maxs
        if not maxs is None:
            self._maxs = np.inf * np.ones(nval)
            maxs, hit = self.__checkvalues__(maxs, True, True)

            if hit:
                raise ValueError(('Expected maxs within [{0}, {1}],' +\
                    ' got {2}'.format(self._mins, self._maxs, maxs)))
            self._maxs = maxs

        else:
            self._maxs = np.inf * np.ones(nval)

        # Set defaults values
        if not defaults is None:
            self._defaults, hit = self.__checkvalues__(defaults, True, True)

            if hit:
                raise ValueError(('Expected defaults within [{0}, {1}],' +\
                    ' got {2}'.format(self._mins, self._maxs, defaults)))
        else:
            self._defaults = np.clip(np.zeros(nval), self._mins, self._maxs)

        # Sampling properties
        self._cov = np.eye(nval, dtype=np.float64)

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
        for i in range(self.nval):
            names.append(dct['data']['name'])
            default.append(dct['data']['default'])
            mins.append(dct['data']['mins'])
            maxs.append(dct['data']['maxs'])
            values.append(dct['data']['values'])

        vect = Vector(names, defaults, mins, maxs)
        vect.values = values

        cov = np.array(dct['cov']).reshape((nval, nval))
        vect.cov = cov

        vect._hitbounds = bool(dct['hitbounds'])

        return vect


    def __str__(self):
            return 'vector ['+', '.join(['{0}:{1:3.3e}'.format(key, self[key]) \
                                    for key in self.names]) + ']'


    def __findname__(self, key):
        idx = np.where(self.names == key)[0]
        if b.shape[0] == 0:
            raise ValueError(('Expected key {0} in the' +
                ' list of names {1}').format(key, self.names))

        return idx


    def __setitem__(self, key, value):
        idx = self.__findname__(key)
        if np.isnan(value):
            raise ValueError('Cannot set value to nan')
        self.values[idx] = value


    def __checkvalues__(self, val, clip=True, hitbounds=False):
        ''' Check vector is of proper length and contains no nan'''

        val = np.atleast_1d(val).flatten().astype(np.float64)

        if val.shape[0] != self.nval:
            raise ValueError('Expected vector of length {0}, got {1}'.format(\
                    self.nval, val.shape[0]))

        if np.any(np.isnan(val)):
            raise ValueError('Cannot process values with NaN')

        hit = False
        if clip:
            if hitbounds:
                hit = np.any((val<self._mins) | (val>self._maxs))

            val = np.clip(val, self._mins, self._maxs)

        return val, hit


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
    def values(self):
        ''' Get data for a given ensemble member set by iens '''
        return self._values


    @values.setter
    def values(self, val):
        ''' Set data for a given ensemble member set by iens '''
        self._values, self._hitbounds = self.__checkvalues__(val, True, True)


    @property
    def cov(self):
        return self._cov


    @cov.setter
    def cov(self, val):
        val = np.atleast_2d(val).astype(np.float64)

        nval = self.nval
        if not val.shape == (nval, nval):
            raise ValueError(('Expected a coviance matrix '+ \
                'of size {0}, got {1}').format((nval, nval), val.shape))

        if not np.allclose(val, val.T):
            raise ValueError('Covariance matrix should be symetric')

        eig, _ = np.linalg.eig(val)
        if np.any(eig<1e-10):
            raise ValueError('Covariance matrix should be semi-definite positive')


    def randomise(self, distribution='normal'):
        ''' Randomise vector data '''

        # Sample vector data
        if distribution == 'normal':
            self.values = np.random.multivariate_normal(self.defaults,
                    self.cov, size=1)

        elif distribution == 'uniform':
            self.values = np.random.uniform(self.min, self.max,
                    (self.nens, self.nval))
        else:
            raise ValueError(('Expected normal or uniform distributoin, ' + \
                    'got {0}').format(distribution))


    def clone(self):
        ''' Clone the vector '''
        clone = Vector(self.names, self.defaults, self.mins, \
                    self.maxs)

        clone.values = self.values
        clone.cov = self.cov

        return clone


    def to_dict(self):
        ''' Write vector data to json format '''

        js = {'nval': self.nval, 'cov':self.cov.tolist(), \
                'hitbounds':self.hitbounds, 'data':[]}

        for i in range(self.nval):
            dd = {'name':self.names[i], \
                'value':self.values[i], \
                'min':self.mins[i], \
                'max':self.maxs[i], \
                'default':self.defaults[i]}
            js['data'].append(dd)

        return js


