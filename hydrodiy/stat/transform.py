''' Module providing various data transforms '''

import math
import sys
import numpy as np
from hydrodiy.data.containers import Vector
from hydrodiy.stat import sutils

__all__ = ['Identity', 'Logit', 'Log', 'BoxCox2', 'BoxCox1',
                'YeoJohnson', 'Reciprocal', 'Softmax', 'AbsLog', \
                'LogSinh']

EPS = 1e-10


def cast(x, y):
    ''' Cast y to the type of x.

        Useful to make sure that a function returns an output that has
        the same type than the input.
    '''
    # Check dtype of inputs if any
    xdtype = x.dtype if hasattr(x, 'dtype') else None
    ydtype = y.dtype if hasattr(y, 'dtype') else None

    # Cast depending on the nature of x and y
    if xdtype is None:
        # x is a basic data type
        # this should work even if y is a
        # 1d or 0d numpy array
        ycast = type(x)(y)

    else:
        # x is a numpy array
        ycast = np.array(y, dtype=xdtype)

    return ycast


def get_transform(name, **kwargs):
    ''' Return instance of transform.
        The function can set parameter values by via kwargs.

        Example:
        >>> BC = get_transform("BoxCox2", lam=0.2)
    '''

    if not name in __all__:
        raise ValueError(('Expected transform name in {0}, '+\
            'got {1}').format(__all__, name))

    trans = getattr(sys.modules[__name__], name)()

    if len(kwargs) > 0:
        for vname in kwargs:
            if not vname in trans.params.names and \
                not vname in trans.constants.names:
                raise ValueError('Expected parameter name to '+\
                    'be in {0} or {1}, got {2}'.format(trans.params.names, \
                        trans.constants.names, vname))

            # Set parameters
            if vname in trans.params.names:
                trans.params[vname] = kwargs[vname]

            # Set constants
            if vname in trans.constants.names:
                trans.constants[vname] = kwargs[vname]

    return trans


class Transform(object):
    ''' Transform base class '''

    def __init__(self, name,\
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


    def _forward(self, x):
        ''' internal forward method (no cast)'''
        raise NotImplementedError('Method _forward not implemented')


    def _backward(self, y):
        ''' internal backward method (no cast)'''
        raise NotImplementedError('Method _backward not implemented')


    def _jacobian_det(self, x):
        ''' internal jacobian_det method (no cast)'''
        raise NotImplementedError('Method _jacobian_det not implemented')


    def forward(self, x):
        ''' Returns the forward transform of x '''
        return cast(x, self._forward(x))


    def backward(self, y):
        ''' Returns the backward transform of y after cast '''
        return cast(y, self._backward(y))


    def jacobian_det(self, x):
        ''' Returns the transformation jacobian_detobian d[forward(x)]/dx
            after cast
        '''
        return cast(x, self._jacobian_det(x))


    def backward_censored(self, y, censor=0.):
        ''' Returns the backward transform of x censored below
            a censor value in the original space.

            This function is useful when dealing with censoring
            in transform space.
        '''
        # Select y to get values above censor only
        # This avoids nan in backward operation
        tcensor = self.forward(censor)
        yc = np.maximum(y, tcensor)

        # Second filtering to ensure that results is
        # equal or greater than  censor
        # (numerical errors of previous operation
        # could compromise that)
        xc = np.maximum(self.backward(yc), censor)

        return xc

    def sample_params(self, nsamples=500, minval=-10., maxval=10.):
        ''' Sample parameters with latin hypercube
            minval and maxval are used to cap the sampling for parameters
            with infinite bounds

        '''
        pmins = self.params.mins
        pmins[np.isinf(pmins)] = minval

        pmaxs = self.params.maxs
        pmaxs[np.isinf(pmaxs)] = maxval

        return sutils.lhs(nsamples, pmins, pmaxs)



class Identity(Transform):
    ''' Identity transform '''

    def __init__(self):
        super(Identity, self).__init__('Identity')

    def _forward(self, x):
        return x

    def _backward(self, y):
        return y

    def _jacobian_det(self, x):
        return np.ones_like(x)



class Logit(Transform):
    ''' Logit transform y = log(x/(1-x))'''

    def __init__(self):
        params = Vector(['lower', 'upper'], \
                    defaults=[0, 1])

        super(Logit, self).__init__('Logit', params)

    def check_upper_lower(self):
        ''' Check that lower <= upper '''
        lower, upper = self.params.values
        if upper <= lower+1e-8:
            raise ValueError('Expected lower<upper, '+\
                'got {0} and {1}'.format(lower, upper))

    def _forward(self, x):
        self.check_upper_lower()
        lower, upper = self.params.values
        value = (x-lower)/(upper-lower)
        return np.log(1./(1-value)-1)

    def _backward(self, y):
        self.check_upper_lower()
        lower, upper = self.params.values
        bnd = 1-1./(1+np.exp(y))
        return bnd*(upper-lower) + lower

    def _jacobian_det(self, x):
        self.check_upper_lower()
        lower, upper = self.params.values
        value = (x-lower)/(upper-lower)
        return  np.where((x > lower+EPS) & (x < upper-EPS), \
                    1./(upper-lower)/value/(1-value), np.nan)

    def sample_params(self, nsamples=500, minval=-10., maxval=10.):
        ''' Sample parameters with latin hypercube '''

        pmins = [minval, 0]
        pmaxs = [maxval, 1]

        samples = sutils.lhs(nsamples, pmins, pmaxs)
        samples[:, 1] = np.sum(samples, axis=1)

        return samples



class Log(Transform):
    ''' Log transform y = log(x+nu) '''

    def __init__(self):
        params = Vector(['nu'], defaults=[1], mins=[EPS])

        super(Log, self).__init__('Log', params)

    def _forward(self, x):
        nu = self.nu
        return np.log(x+nu)

    def _backward(self, y):
        nu = self.nu
        return np.exp(y)-nu

    def _jacobian_det(self, x):
        nu = self.nu
        return np.where(x+nu > EPS, 1./(x+nu), np.nan)

    def sample_params(self, nsamples=500, minval=-6., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, nsamples)
        samples = np.exp(samples)[:, None]

        return samples



class BoxCox2(Transform):
    ''' BoxCox transform with 2 parameters y = ((nu+x)^lambda-1)/lambda '''

    def __init__(self):
        params = Vector(['nu', 'lam'], [0., 1.], \
                    [EPS, -3.], [np.inf, 3.])

        super(BoxCox2, self).__init__('BoxCox2', params)

    def _forward(self, x):
        nu, lam = self.params.values
        if abs(lam) > EPS:
            return (np.exp(np.log(x+nu)*lam)-1)/lam
        else:
            return np.log(x+nu)

    def _backward(self, y):
        nu, lam = self.params.values

        if abs(lam) > EPS:
            u = lam*y+1
            return np.exp(np.log(u)/lam)-nu
        else:
            return np.exp(y)-nu

    def _jacobian_det(self, x):
        nu, lam = self.params.values

        if abs(lam) > EPS:
            return np.where(x+nu > EPS, \
                    np.exp(np.log(x+nu)*(lam-1.)), np.nan)
        else:
            return np.where(x+nu > EPS, 1./(x+nu), np.nan)

    def sample_params(self, nsamples=500, minval=-6., maxval=0.):
        pmins = [minval, 0]
        pmaxs = [maxval, 1]

        # Generate parameters samples in log space
        samples = sutils.lhs(nsamples, pmins, pmaxs)
        samples[:, 0] = np.exp(samples[:, 0])

        return samples



class BoxCox1(Transform):
    ''' BoxCox transform y = ((x0+x)^lambda-1)/lambda
        with normalisation constant x0 = max(x)/5
    '''
    def __init__(self):
        params = Vector(['lam'], [1.], [-3.], [3.])

        # Define the nu constant and set it to inf by default
        # to force proper setup
        constants = Vector(['x0'], [np.inf], [EPS], [np.inf])

        super(BoxCox1, self).__init__('BoxCox1', params, constants)
        self.BC = BoxCox2()

    def _forward(self, x):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')
        self.BC.params.values = [x0, self.params.lam]
        return self.BC.forward(x)

    def _backward(self, y):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')
        self.BC.params.values = [x0, self.params.lam]
        return self.BC.backward(y)

    def _jacobian_det(self, x):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')
        self.BC.params.values = [x0, self.params.lam]
        return self.BC.jacobian_det(x)

    def sample_params(self, nsamples=500, minval=0., maxval=1.):
        return np.random.uniform(minval, maxval, size=nsamples)[:, None]



class YeoJohnson(Transform):
    ''' YeoJohnson transform '''

    def __init__(self):
        params = Vector(['nu', 'scale', 'lam'],\
            [0., 1., 1.], [-np.inf, 1e-5, -1.],\
            [np.inf, np.inf, 3.])

        super(YeoJohnson, self).__init__('YeoJohnson', params)

    def _forward(self, x):
        nu, scale, lam = self.params.values
        x = np.atleast_1d(x)
        y = x*np.nan
        w = nu+x*scale
        ipos = w >= EPS

        if not np.isclose(lam, 0.0):
            y[ipos] = (np.power(w[ipos]+1, lam)-1)/lam

        if np.isclose(lam, 0.0):
            y[ipos] = np.log(w[ipos]+1)

        if not np.isclose(lam, 2.0):
            y[~ipos] = -(np.power(-w[~ipos]+1, 2-lam)-1)/(2-lam)

        if np.isclose(lam, 2.0):
            y[~ipos] = -np.log(-w[~ipos]+1)

        return y

    def _backward(self, y):
        nu, scale, lam = self.params.values
        y = np.atleast_1d(y)
        x = y*np.nan
        ipos = y >= EPS

        if not np.isclose(lam, 0.0):
            x[ipos] = np.power(lam*y[ipos]+1, 1./lam)-1
        else:
            x[ipos] = np.exp(y[ipos])-1

        if not np.isclose(lam, 2.0):
            x[~ipos] = -np.power(-(2-lam)*y[~ipos]+1, 1./(2-lam))+1
        else:
            x[~ipos] = -np.exp(-y[~ipos])+1

        return (x-nu)/scale

    def _jacobian_det(self, x):
        nu, scale, lam = self.params.values
        x = np.atleast_1d(x)
        j = x*np.nan
        w = nu+x*scale
        ipos = w >= EPS

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
    ''' LogSinh transform with normalisation constant
        x0 = max(x)/5
        y=1/b*log(sinh(a+b*x/x0))
    '''

    def __init__(self):
        params = Vector(['a', 'b'], [0., 1.], [EPS, EPS])

        # Define the nu constant and set it to inf by default
        # to force proper setup
        constants = Vector(['x0'], [np.inf], [EPS], [np.inf])

        super(LogSinh, self).__init__('LogSinh', \
            params, constants)

    def _forward(self, x):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')

        a, b = self.params.values
        xn = x/x0
        w = a + b*xn
        return np.where(xn > -a/b+EPS, \
            (w+np.log((1.-np.exp(-2.*w))/2.))/b, np.nan)

    def _backward(self, y):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')

        a, b = self.params.values
        w = b*y
        return x0*(y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b)

    def _jacobian_det(self, x):
        x0 = self.constants.x0
        if np.isinf(x0):
            raise ValueError('x0 is inf. It must be set to a proper value')

        a, b = self.params.values
        xn = x/x0
        w = a+b*xn
        return 1./x0*np.where(xn > -a/b+EPS, 1./np.tanh(w), np.nan)

    def sample_params(self, nsamples=500, loga_scale=3., logb_sig=1.):
        ''' Sample from informative prior '''

        # Generate parameters samples from informative prior
        loga = -np.random.exponential(scale=loga_scale, size=nsamples)
        loga = np.maximum(loga, math.log(self.params.mins[0]))

        logb = np.random.normal(loc=0., scale=logb_sig, size=nsamples)
        logb = np.maximum(logb, math.log(self.params.mins[1]))

        samples = np.exp(np.column_stack([loga, logb]))

        return samples



class Reciprocal(Transform):
    ''' Reciprocal transform y=-1/(nu+x) '''

    def __init__(self):
        params = Vector(['nu'], [1.], [EPS])

        super(Reciprocal, self).__init__('Reciprocal', params)

    def _forward(self, x):
        nu = self.params.values
        return np.where(x > -nu, -1./(nu+x), np.nan)

    def _backward(self, y):
        nu = self.params.values
        return np.where(y < -EPS, -1./y-nu, np.nan)

    def _jacobian_det(self, x):
        nu = self.params.values
        return np.where(x > -nu, 1./(nu+x)**2, np.nan)

    def sample_params(self, nsamples=500, minval=-7., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, nsamples)
        samples = np.exp(samples)[:, None]

        return samples



class Softmax(Transform):
    ''' Softmax transform '''

    def __init__(self):
        super(Softmax, self).__init__('Softmax')


    def _forward(self, x):
        # Check inputs
        x = np.atleast_2d(x)
        if x.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(x.ndim))

        if np.any(x < 0):
            raise ValueError('x < 0')

        # Sum along columns
        sx = np.sum(x, axis=1)
        if np.any(sx > 1-EPS):
            raise ValueError('sum(x) >= 1')

        return np.log(x/(1-sx[:, None]))

    def _backward(self, y):
        # Check inputs
        y = np.atleast_2d(y)
        if y.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(y.ndim))

        # Back transform
        x = np.exp(y)
        return x/(1+np.sum(x, axis=1)[:, None])

    def _jacobian_det(self, x):
        ''' See
        https://en.wikipedia.org/wiki/Determinant#Sylvester.27s_determinant_theorem
        '''
        # Check inputs
        x = np.atleast_2d(x)
        if x.ndim > 2:
            raise ValueError('Expected ndim 2, got {0}'.format(x.ndim))

        if np.any(x < 0):
            raise ValueError('x < 0')

        sx = np.sum(x, axis=1)
        if np.any(sx > 1-EPS):
            raise ValueError('sum(x) >= 1')

        px = np.prod(x, axis=1)
        return (1+sx/(1-sx))/px



class AbsLog(Transform):
    ''' Absolute log transform y=sign(x) * log(cst+abs(x))-log(cst) '''

    def __init__(self):
        params = Vector(['nu'], [1e-10], [1e-10], [np.inf])
        super(AbsLog, self).__init__('AbsLog', params)

    def _forward(self, x):
        nu = self.params.values
        return np.sign(x)*(np.log(nu+np.abs(x))-math.log(nu))

    def _backward(self, y):
        nu = self.params.values
        return np.sign(y)*(np.exp(np.abs(y)+math.log(nu))-nu)

    def _jacobian_det(self, x):
        nu = self.params.values
        return 1./(nu+np.abs(x))

    def sample_params(self, nsamples=500, minval=-7., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, nsamples)
        samples = np.exp(samples)[:, None]

        return samples



