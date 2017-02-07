import math
import numpy as np
from hydrodiy.data.containers import Vector

__all__ = ['Identity', 'Logit', 'Log', 'BoxCox', 'YeoJohnson', \
                'LogSinh', 'Reciprocal']

EPS = 1e-10


class Transform(object):
    ''' Transform base class '''

    def __init__(self, name,
        pnames=None, \
        defaults=None, \
        mins=None, \
        maxs=None,
        constants=None):
        """
        Create instance of a data transform object

        Parameters
        -----------
        name : str
            Name of transformation
        pnames : list
            Names of transform parameters
        defaults : list
            Defaults values for parameters
        mins : list
            Minum values for parameters
        maxs : list
            Maximum values for parameters
        constants : list
            Name of constants

        """
        self.name = name
        self._params = Vector(pnames, defaults, mins, maxs)
        self._constants = Vector(constants)


    def __setitem__(self, key, value):
        if self._constants.nval == 0:
            self._params[key] = value
        else:
            if key in self._params.names:
                self._params[key] = value
            else:
                self._constants[key] = value


    def __getitem__(self, key):
        if self._constants.nval == 0:
            return self._params[key]
        else:
            if key in self._params.names:
                return self._params[key]
            else:
                return self._constants[key]


    @property
    def nparams(self):
        return self._params.nval


    @property
    def hitbounds(self):
        return self._params.hitbounds


    @property
    def pnames(self):
        ''' Get parameter pnames '''
        return self._params.names


    @property
    def mins(self):
        ''' Get parameter minimum bounds '''
        return self._params.mins


    @property
    def maxs(self):
        ''' Get parameter maximum bounds '''
        return self._params.maxs


    @property
    def defaults(self):
        ''' Get parameter defaults '''
        return self._params.defaults


    @property
    def params(self):
        ''' Get parameters '''
        return self._params.values


    @params.setter
    def params(self, values):
        ''' Set parameters '''
        self._params.values = values


    def reset(self):
        ''' Reset parameter values '''
        self._params.reset()


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
                pnames=['lower', 'upper'],
                defaults=[0, 1])

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
                pnames=['shift'],
                defaults=1., \
                mins=[EPS])

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
            pnames=['shift', 'lambda'],
            defaults=[0., 1.], \
            mins=[EPS, EPS], \
            maxs=[np.inf, 3.])

    def forward(self, x):
        shift, lam = self.params
        return (np.power(x+shift, lam)-1)/lam

    def backward(self, y):
        shift, lam = self.params
        u = lam*y+1
        return np.power(u, 1./lam)-shift

    def jacobian_det(self, x):
        shift, lam = self.params
        return np.where(x+shift>EPS, np.power(x+shift, lam-1.), np.nan)



class YeoJohnson(Transform):

    def __init__(self):
        Transform.__init__(self, 'YeoJohnson',
            pnames=['shift', 'scale', 'lambda'],
            defaults=[0., 1., 1.], \
            mins=[-np.inf, 1e-5, -1.], \
            maxs=[np.inf, np.inf, 3.])

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
            pnames=['a', 'b'], \
            defaults=[0., 1.], \
            mins=[-np.inf, EPS])

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
            pnames=['shift'], \
            defaults=[1.], \
            mins=[EPS])

    def forward(self, x):
        shift = self.params
        return np.where(x>-shift, 1./(shift+x), np.nan)

    def backward(self, y):
        shift = self.params
        return np.where(y>EPS, 1./y-shift, np.nan)

    def jacobian_det(self, x):
        shift = self.params
        return np.where(x>-shift, 1./(shift+x)**2, np.nan)


