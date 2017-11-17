import math
import sys
import numpy as np
from hydrodiy.data.containers import Vector

__all__ = ['Identity', 'Logit', 'Log', 'BoxCox', 'YeoJohnson', \
                'LogSinh', 'Reciprocal', 'Softmax']

EPS = 1e-10


def get_transform(name, **kwargs):
    ''' Return instance of transform.
        The function can set parameter values by via kwargs.

        Example:
        >>> BC = get_transform("BoxCox", lam=0.2)
    '''

    if not name in __all__:
        raise ValueError(('Expected transform name in {0}, '+\
            'got {1}').format(__all__, name))

    trans = getattr(sys.modules[__name__], name)()

    if len(kwargs)>0:
        # Set parameters
        for pname in kwargs:
            if not pname in trans.params.names:
                raise ValueError('Expected parameter name to '+\
                    'be in {0}, got {1}'.format(trans.params.names, \
                        pname))
            setattr(trans, pname, kwargs[pname])

    return trans


class Transform(object):
    ''' Transform base class '''

    def __init__(self, name,
        params=Vector([]), \
        constants=Vector([])):
        """
        Create instance of a data transform object

        Parameters
        -----------
        name : str
            Name of transformation
        params : hydrodiy.data.containers.Vector
            Parameter vector
        constants : hydrodiy.data.containers.Vector
            Consant vector

        """
        self.name = name
        self._params = params
        self._constants = constants


    def __getattribute__(self, name):
        # Except name, _params and _constants to avoid infinite recursion
        if name in ['name', '_params', '_constants']:
            return super(Transform, self).__getattribute__(name)

        if name in self._params.names:
            return getattr(self._params, name)

        if name in self._constants.names:
            return getattr(self._constants, name)

        return super(Transform, self).__getattribute__(name)


    def __setattr__(self, name, value):
        # Except name, _params and _constants to avoid infinite recursion
        if name in ['name', '_params', '_constants']:
            super(Transform, self).__setattr__(name, value)
            return

        if name in self._params.names:
            setattr(self._params, name, value)

        elif name in self._constants.names:
            setattr(self._constants, name, value)

        else:
            super(Transform, self).__setattr__(name, value)


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

    def __str__(self):
        str = '{0} transform:\n'.format(self.name)
        str += '\tparams    : {0}\n'.format(self.params.names)
        str += '\tconstants : {0}\n'.format(self.constants.names)

        return str


    @property
    def params(self):
        ''' Get parameters '''
        return self._params


    @property
    def constants(self):
        ''' Get constants '''
        return self._constants


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


    def backward_censored(self, y, censor=0.):
        ''' Returns the backward transform of x censored below
            a censor value in the original space.

            This function is useful when dealing with censoring
            in transform space.
        '''
        tcensor = self.forward(censor)
        yc = np.maximum(y, tcensor)
        return self.backward(yc)


class Identity(Transform):

    def __init__(self):
        super(Identity, self).__init__('Identity')

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def jacobian_det(self, x):
        return np.ones_like(x)



class Logit(Transform):

    def __init__(self):
        params = Vector(['lower', 'upper'], \
                    defaults=[0, 1])

        super(Logit, self).__init__('Logit', params)


    def forward(self, x):
        lower, upper = self.params.values
        value = (x-lower)/(upper-lower)
        return np.log(1./(1-value)-1)

    def backward(self, y):
        lower, upper = self.params.values
        bnd = 1-1./(1+np.exp(y))
        return bnd*(upper-lower) + lower

    def jacobian_det(self, x):
        lower, upper = self.params.values
        value = (x-lower)/(upper-lower)
        return  np.where((x>lower+EPS) & (x<upper-EPS), \
                    1./(upper-lower)/value/(1-value), np.nan)



class Log(Transform):

    def __init__(self):
        params = Vector(['shift'], defaults=[1], mins=[EPS])

        super(Log, self).__init__('Log', params)


    def forward(self, x):
        shift = self.shift
        return np.log(x+shift)

    def backward(self, y):
        shift = self.shift
        return np.exp(y)-shift

    def jacobian_det(self, x):
        shift = self.shift
        return np.where(x+shift>EPS, 1./(x+shift), np.nan)



class BoxCox(Transform):

    def __init__(self):
        params = Vector(['shift', 'lam'], [0., 1.], \
                    [EPS, -3.], [np.inf, 3.])

        super(BoxCox, self).__init__('BoxCox', params)


    def forward(self, x):
        shift, lam = self.params.values
        if abs(lam)>EPS:
            return (np.exp(np.log(x+shift)*lam)-1)/lam
        else:
            return np.log(x+shift)

    def backward(self, y):
        shift, lam = self.params.values

        if abs(lam)>EPS:
            u = lam*y+1
            return np.exp(np.log(u)/lam)-shift
        else:
            return np.exp(y)-shift


    def jacobian_det(self, x):
        shift, lam = self.params.values

        if abs(lam)>EPS:
            return np.where(x+shift>EPS, np.exp(np.log(x+shift)*(lam-1.)), np.nan)
        else:
            return np.where(x+shift>EPS, 1./(x+shift), np.nan)



class YeoJohnson(Transform):

    def __init__(self):
        params = Vector(['shift', 'scale', 'lambda'],\
            [0., 1., 1.], [-np.inf, 1e-5, -1.],\
            [np.inf, np.inf, 3.])

        super(YeoJohnson, self).__init__('YeoJohnson', params)


    def forward(self, x):
        shift, scale, lam = self.params.values
        x = np.atleast_1d(x)
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
        shift, scale, lam = self.params.values
        y = np.atleast_1d(y)
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
        shift, scale, lam = self.params.values
        x = np.atleast_1d(x)
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
        params = Vector(['a', 'b'], [0., 1.], [EPS, EPS])

        super(LogSinh, self).__init__('LogSinh', params)


    def forward(self, x):
        a, b = self.params.values
        w = a + b*x
        return np.where(x>-a/b+EPS, (w+np.log((1.-np.exp(-2.*w))/2.))/b, np.nan)

    def backward(self, y):
        a, b = self.params.values
        w = b*y
        return y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b

    def jacobian_det(self, x):
        a, b = self.params.values
        w = a + b*x
        return np.where(x>-a/b+EPS, 1./np.tanh(w), np.nan)


class Reciprocal(Transform):

    def __init__(self):
        params = Vector(['shift'], [1.], [EPS])

        super(Reciprocal, self).__init__('Reciprocal', params)

    def forward(self, x):
        shift = self.params.values
        return np.where(x>-shift, -1./(shift+x), np.nan)

    def backward(self, y):
        shift = self.params.values
        return np.where(y<-EPS, -1./y-shift, np.nan)

    def jacobian_det(self, x):
        shift = self.params.values
        return np.where(x>-shift, 1./(shift+x)**2, np.nan)


class Softmax(Transform):

    def __init__(self):
        super(Softmax, self).__init__('Softmax')


    def forward(self, x):
        # Check inputs
        x = np.atleast_2d(x)
        if x.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(x.ndim))

        if np.any(x<0):
            raise ValueError('x < 0')

        # Sum along columns
        sx = np.sum(x, axis=1)
        if np.any(sx>1-EPS):
            raise ValueError('sum(x) >= 1')

        return np.log(x/(1-sx[:, None]))

    def backward(self, y):
        # Check inputs
        y = np.atleast_2d(y)
        if y.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(y.ndim))

        # Back transform
        x = np.exp(y)
        return x/(1+np.sum(x, axis=1)[:, None])

    def jacobian_det(self, x):
        ''' See
        https://en.wikipedia.org/wiki/Determinant#Sylvester.27s_determinant_theorem
        '''
        # Check inputs
        x = np.atleast_2d(x)
        if x.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(x.ndim))

        if np.any(x<0):
            raise ValueError('x < 0')

        sx = np.sum(x, axis=1)
        if np.any(sx>1-EPS):
            raise ValueError('sum(x) >= 1')

        px = np.prod(x, axis=1)
        return (1+sx/(1-sx))/px



