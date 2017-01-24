import math
import numpy as np

__all__ = ['Identity', 'Logit', 'Log', 'BoxCox', 'YeoJohnson', \
                'LogSinh', 'Reciprocal']

EPS = 1e-10


class Parameterised(object):
    ''' Simple interface to common transform functions '''

    def __init__(self, name,
            params_names=None, \
            params_default=None, \
            params_mins=None, \
            params_maxs=None,
            object_type_name='Object'):
        ''' Initialise object with name, number of parameters (nparams),
        default, min and max value for parameters, and object name.
        '''
        self._name = name
        self._object_type_name = object_type_name

        # Param names and number of params
        if params_names is None:
            nparams = 0
            params_names = np.array([])
        else:
            params_names = np.atleast_1d(params_names).astype(str)
            nparams = len(params_names)

            if nparams != len(np.unique(params_names)):
                raise ValueError('Non unique parameter names {0}'.format(\
                    params_names))

        self._nparams = nparams
        self._params_names = params_names

        # Initialise params
        self._params = np.repeat(np.nan, nparams)

        # Set default params
        if params_default is None:
            params_default = np.zeros(nparams)

        self._params_default = np.atleast_1d(params_default).astype(np.float64)
        if len(self._params_default) != nparams:
            raise ValueError(('Expected params_default of length' + \
                ' {0}, got {1}').format(nparams, \
                    self._params_default.shape[0]))

        # Set min params
        if params_mins is None:
            params_mins = np.repeat(-np.inf, nparams)

        self._params_mins = np.atleast_1d(params_mins).astype(np.float64)
        if len(self._params_mins) != nparams:
            raise ValueError(('Expected params_mins of length' + \
                ' {0}, got {1}').format(nparams, \
                    self._params_mins.shape[0]))

        # Set max params
        if params_maxs is None:
            params_maxs = np.repeat(np.inf, nparams)

        self._params_maxs = np.atleast_1d(params_maxs).astype(np.float64)
        if len(self._params_maxs) != nparams:
            raise ValueError(('Expected params_maxs of length' + \
                ' {0}, got {1}').format(nparams, \
                    self._params_maxs.shape[0]))

    def __str__(self):
        s = '\n{0} {1}\n'.format(self._name, self._object_type_name)
        if self._nparams > 0:
            s += '  Params = [' + ', '.join(['{0}:{1:3.3e}'.format(nm, p) \
                for nm, p in  zip(self._params_names, self._params)]) + ']'
        s += '\n'
        return s

    def __setitem__(self, key, item):
        item = float(item)
        if np.isnan(item):
            raise ValueError('Cannot set parameter with nan')

        idx = self.params_names == key
        mini = self.params_mins[idx]
        maxi = self.params_maxs[idx]
        self._params[idx] = np.clip(item, mini, maxi)

    def __getitem__(self, key):
        return self._params[self.params_names == key][0]

    @property
    def name(self):
        return self._name

    @property
    def nparams(self):
        return self._nparams

    @property
    def params_names(self):
        return self._params_names

    @property
    def params_default(self):
        return self._params_default

    @property
    def params_mins(self):
        return self._params_mins

    @property
    def params_maxs(self):
        return self._params_maxs

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        if self.nparams == 0:
            return

        value = np.atleast_1d(value).astype(np.float64)

        # Check values
        if len(value) != self._nparams:
            raise ValueError(('Expected params of length' + \
                ' {0}, got {1}').format(self.nparams, \
                    value.shape[0]))

        if np.any(np.isnan(value)):
            raise ValueError('Cannot set parameters with a nan value ({0})'.format(\
                value))

        # Allocate parameters
        self._params = np.clip(value, self.params_mins, self.params_maxs)


    def reset(self):
        ''' Reset parameter values to default '''
        self.params = self.params_default



class Transform(Parameterised):
    ''' Transform base class '''

    def __init__(self, name,
        params_names=None, \
        params_default=None, \
        params_mins=None, \
        params_maxs=None):

        Parameterised.__init__(self, name, params_names, \
            params_default, params_mins, \
            params_maxs, 'Transform')


    def forward(self, x):
        ''' Returns the forward transform of x '''
        raise NotImplementedError('Method forward not implemented')

    def backward(self, y):
        ''' Returns the backward transform of x '''
        raise NotImplementedError('Method backward not implemented')

    def jacobian_det(self, x):
        ''' Returns the transformation jacobian_detobian d[forward(x)]/dx '''
        raise NotImplementedError('Method jacobian_det not implemented')



class Identity(Transform):

    def __init__(self):
        Transform.__init__(self, 'Identity')

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def jacobian_det(self, x):
        return np.ones_like(x)



class Logit(Transform):

    def __init__(self):
        Transform.__init__(self, 'Logit', \
                params_names=['lower', 'upper'],
                params_default=[0, 1])

    def forward(self, x):
        lower, upper = self.params
        value = (x-lower)/(upper-lower)
        return np.log(1./(1-value)-1)

    def backward(self, y):
        lower, upper = self.params
        bnd = 1-1./(1+np.exp(y))
        return bnd*(upper-lower) + lower

    def jacobian_det(self, x):
        lower, upper = self.params
        value = (x-lower)/(upper-lower)
        return  np.where((x>lower+EPS) & (x<upper-EPS), \
                    1./(upper-lower)/value/(1-value), np.nan)



class Log(Transform):

    def __init__(self):
        Transform.__init__(self, 'Log', \
                params_names=['shift'],
                params_default=1., \
                params_mins=[EPS])

    def forward(self, x):
        shift = self.params[0]
        return np.log(x+shift)

    def backward(self, y):
        shift = self.params[0]
        return np.exp(y)-shift

    def jacobian_det(self, x):
        shift = self.params[0]
        return np.where(x+shift>EPS, 1./(x+shift), np.nan)



class BoxCox(Transform):

    def __init__(self):
        Transform.__init__(self, 'BoxCox',
            params_names=['shift', 'lambda'],
            params_default=[0., 1.], \
            params_mins=[EPS, EPS], \
            params_maxs=[np.inf, 3.])

    def forward(self, x):
        shift, lam = self.params
        return (np.power(x+shift, lam)-shift**lam)/lam

    def backward(self, y):
        shift, lam = self.params
        u = lam*y+shift**lam
        return np.power(u, 1./lam)-shift

    def jacobian_det(self, x):
        shift, lam = self.params
        return np.where(x+shift>EPS, np.power(x+shift, lam-1.), np.nan)



class YeoJohnson(Transform):

    def __init__(self):
        Transform.__init__(self, 'YeoJohnson',
            params_names=['shift', 'scale', 'lambda'],
            params_default=[0., 1., 1.], \
            params_mins=[-np.inf, 1e-5, -1.], \
            params_maxs=[np.inf, np.inf, 3.])

    def forward(self, x):
        shift, scale, lam = self.params
        y = x*np.nan
        w = shift+x*scale
        ipos = w>=EPS

        if not np.isclose(lam, 0.0):
            y[ipos] = (np.power(w[ipos]+1, lam)-1)/lam

        if np.isclose(lam, 0.0):
            y[ipos] = np.log(w[ipos]+1)

        if not np.isclose(lam, 2.0):
            y[~ipos] = -(np.power(-w[~ipos]+1, 2-lam)-1)/(2-lam)

        if np.isclose(lam, 2.0):
            y[~ipos] = -np.log(-w[~ipos]+1)

        return y


    def backward(self, y):
        shift, scale, lam = self.params
        x = y*np.nan
        ipos = y>=EPS

        if not np.isclose(lam, 0.0):
            x[ipos] = np.power(lam*y[ipos]+1, 1./lam)-1
        else:
            x[ipos] = np.exp(y[ipos])-1

        if not np.isclose(lam, 2.0):
            x[~ipos] = -np.power(-(2-lam)*y[~ipos]+1, 1./(2-lam))+1
        else:
            x[~ipos] = -np.exp(-y[~ipos])+1

        return (x-shift)/scale


    def jacobian_det(self, x):
        shift, scale, lam = self.params
        j = x*np.nan
        w = shift+x*scale
        ipos = w>=EPS

        if not np.isclose(lam, 0.0):
            j[ipos] = (w[ipos]+1)**(lam-1)

        if np.isclose(lam, 0.0):
            j[ipos] = 1/(w[ipos]+1)

        if not np.isclose(lam, 2.0):
            j[~ipos] = (-w[~ipos]+1)**(1-lam)

        if np.isclose(lam, 2.0):
            j[~ipos] = 1/(-w[~ipos]+1)

        return j*scale


class LogSinh(Transform):

    def __init__(self):
        Transform.__init__(self, 'LogSinh', \
            params_names=['a', 'b'], \
            params_default=[0., 1.], \
            params_mins=[-np.inf, EPS])

    def forward(self, x):
        a, b = self.params
        w = a + b*x
        return np.where(x>-a/b+EPS, (w+np.log((1.-np.exp(-2.*w))/2.))/b, np.nan)

    def backward(self, y):
        a, b = self.params
        w = b*y
        return y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b

    def jacobian_det(self, x):
        a, b = self.params
        w = a + b*x
        return np.where(x>-a/b+EPS, 1./np.tanh(w), np.nan)


class Reciprocal(Transform):

    def __init__(self):
        Transform.__init__(self, 'Reciprocal', \
            params_names=['shift'], \
            params_default=[1.], \
            params_mins=[EPS])

    def forward(self, x):
        shift = self.params
        return np.where(x>-shift, 1./(shift+x), np.nan)

    def backward(self, y):
        shift = self.params
        return np.where(y>EPS, 1./y-shift, np.nan)

    def jacobian_det(self, x):
        shift = self.params
        return np.where(x>-shift, 1./(shift+x)**2, np.nan)


