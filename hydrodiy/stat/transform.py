''' Module providing various data transforms '''

import math
import sys
import numpy as np
import inspect
from hydrodiy import PYVERSION

from scipy.stats import norm

from hydrodiy.data.containers import Vector
from hydrodiy.data import dutils
from hydrodiy.stat import sutils

__all__ = ['Identity', 'Logit', 'Log', 'BoxCox2', \
                'BoxCox1lam','BoxCox1nu', \
                'YeoJohnson', 'Reciprocal', 'Softmax', 'Sinh', \
                'LogSinh', 'Manly']

# Constant to detect zeros
EPS = 1e-10

def _class_constructor_argnames(classobj):
    ''' Get class constructor arguments '''
    if PYVERSION == 3:
        sign = inspect.signature(classobj)
        argnames = sign.parameters
    elif PYVERSION == 2:
        argnames = inspect.getargspec(classobj.__init__).args

    return argnames


def get_transform(name, **kwargs):
    ''' Return instance of transform.
        The function can set mininu (mininum of the threshold parameter)
        and parameter values by via kwargs.

        Example:
        >>> BC = get_transform("BoxCox2", lam=0.2)
    '''

    if not name in __all__:
        raise ValueError(('Expected transform name in {0}, '+\
            'got {1}').format(__all__, name))

    # Get instance of transform, setting the constructor arguments
    trans_class = getattr(sys.modules[__name__], name)
    constructor_args = _class_constructor_argnames(trans_class)
    class_kwargs = {}
    topop = []
    for key in kwargs:
        if key in constructor_args:
            class_kwargs[key] = kwargs[key]
            topop.append(key)

    for key in topop:
        kwargs.pop(key)

    trans = trans_class(**class_kwargs)

    # Set parameters
    if len(kwargs) > 0:
        for vname in kwargs:
            if not vname in trans.params.names and \
                    not vname in trans.constants.names:
                continue

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
        return dutils.cast(x, self._forward(x))


    def backward(self, y):
        ''' Returns the backward transform of y after cast '''
        return dutils.cast(y, self._backward(y))


    def jacobian_det(self, x):
        ''' Returns the transformation jacobian_detobian d[forward(x)]/dx
            after cast
        '''
        return dutils.cast(x, self._jacobian_det(x))


    def backward_censored(self, y, censor=0.):
        ''' Returns the backward transform of x censored below
            a censor value in the original space.

            This function is useful when dealing with censoring
            in transform space.
        '''
        # Select y to get values above censor only
        # This avoids nan in backward operation
        tcensor = self.forward(censor)
        if np.isnan(tcensor):
            yc = y
        else:
            yc = np.maximum(y, tcensor)

        # Second filtering to ensure that results is
        # equal or greater than  censor
        # (numerical errors of previous operation
        # could compromise that)
        xc = np.maximum(self.backward(yc), censor)

        return xc

    def params_sample(self, nsamples=500, minval=-10., maxval=10.):
        ''' Sample parameters with latin hypercube
            minval and maxval are used to cap the sampling for parameters
            with infinite bounds

        '''
        pmins = self.params.mins
        pmins[np.isinf(pmins)] = minval

        pmaxs = self.params.maxs
        pmaxs[np.isinf(pmaxs)] = maxval

        return sutils.lhs(nsamples, pmins, pmaxs)


    def params_logprior(self):
        ''' Flat log prior for transform parameters '''
        val = self.params.values
        mins = self.params.mins
        maxs = self.params.maxs

        ck = np.all((val-mins>=0) & (maxs-val>=0))
        return 0 if ck else -np.inf



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
        params = Vector(['lower', 'logdelta'], \
                    [0., 0.], [-np.inf, -10], [np.inf, 10])

        super(Logit, self).__init__('Logit', params)

    def _forward(self, x):
        lower, logdelta = self.params.values
        upper = lower + math.exp(logdelta)
        value = (x-lower)/(upper-lower)
        return np.log(1./(1-value)-1)

    def _backward(self, y):
        lower, logdelta = self.params.values
        upper = lower + math.exp(logdelta)
        bnd = 1-1./(1+np.exp(y))
        return bnd*(upper-lower) + lower

    def _jacobian_det(self, x):
        lower, logdelta = self.params.values
        upper = lower + math.exp(logdelta)
        value = (x-lower)/(upper-lower)
        return  np.where((x > lower+EPS) & (x < upper-EPS), \
                    1./(upper-lower)/value/(1-value), np.nan)

    def params_sample(self, nsamples=500, minval=-10., maxval=10.):
        ''' Sample parameters with latin hypercube '''

        pmins = [-10, self.params.mins[1]]
        pmaxs = [10, self.params.maxs[1]]

        samples = sutils.lhs(nsamples, pmins, pmaxs)

        return samples


class Log(Transform):
    ''' Log transform y = log(x+nu) '''

    def __init__(self, mininu=EPS, base=None):
        params = Vector(['nu'], defaults=[mininu], mins=[mininu])
        super(Log, self).__init__('Log', params)
        self.mininu = mininu

        if base is None:
            self.basefactor = 1
        else:
            self.basefactor = math.log(base)

    def _forward(self, x):
        nu = self.params.values[0]
        bf = self.basefactor
        return np.log(x+nu)/bf

    def _backward(self, y):
        nu = self.params.values[0]
        bf = self.basefactor
        return np.exp(bf*y)-nu

    def _jacobian_det(self, x):
        nu = self.params.values[0]
        bf = self.basefactor
        return np.where(x+nu > self.mininu, 1./(x+nu)/bf, np.nan)

    def params_sample(self, nsamples=500, minval=-6., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, nsamples)
        samples = np.exp(samples)[:, None]+self.mininu

        return samples


class BoxCox2(Transform):
    ''' BoxCox transform with 2 parameters y = ((nu+x)^lambda-1)/lambda '''

    def __init__(self, mininu=EPS, minilam=0.):
        ''' Instanciate the BoxCox transform with two parameters

        Parameters
        -----------
        mininu : float
            Minimum of the shift parameter (nu)
        minilamprior : float
            Mininum of the value allowed for lambda in prior probability
        '''
        if minilam < -3:
            raise ValueError('Expected minilam > -3, got {0}'.format(minilam))

        params = Vector(['nu', 'lam'], [mininu, 1.], \
                    [mininu, minilam], [np.inf, 3.])

        super(BoxCox2, self).__init__('BoxCox2', params)
        self.mininu = mininu

    def _forward(self, x):
        nu, lam = self.params.values
        if abs(lam) > EPS:
            return (np.power(x+nu, lam)-1)/lam
        else:
            return np.log(x+nu)

    def _backward(self, y):
        nu, lam = self.params.values

        if abs(lam) > EPS:
            u = lam*y+1
            return np.power(u, 1./lam)-nu
        else:
            return np.exp(y)-nu

    def _jacobian_det(self, x):
        nu, lam = self.params.values

        if abs(lam) > EPS:
            return np.where(x+nu > self.mininu, \
                    np.power(x+nu, lam-1.), np.nan)
        else:
            return np.where(x+nu > self.mininu, 1./(x+nu), np.nan)

    def params_sample(self, nsamples=500, minval=-6., maxval=0.):
        pmins = [minval, 0]
        pmaxs = [maxval, 1]

        # Generate parameters samples in log space
        samples = sutils.lhs(nsamples, pmins, pmaxs)
        samples[:, 0] = np.exp(samples[:, 0])+self.mininu

        return samples


class BoxCox1lam(Transform):
    ''' boxcox transform y = ((nu+x)^lambda-1)/lambda
        with shift parameter nu fixed
    '''
    def __init__(self, mininu=EPS, minilam=0.):
        params = Vector(['lam'], [1.], [minilam], [3.])

        # define the nu constant and set it to inf by default
        # to force proper setup
        constants = Vector(['nu'], [np.nan], [mininu], [np.inf], \
                                        accept_nan=True)

        super(BoxCox1lam, self).__init__('BoxCox1lam', params, constants)
        self.BC = BoxCox2(mininu=mininu, minilam=minilam)

    def get_nu(self):
        nu = self.constants.values[0]
        if np.isnan(nu):
            raise ValueError('nu is nan. It must be set to a proper value')
        return nu

    def _forward(self, x):
        self.BC.params.values = [self.get_nu(), self.params.values[0]]
        return self.BC.forward(x)

    def _backward(self, y):
        self.BC.params.values = [self.get_nu(), self.params.values[0]]
        return self.BC.backward(y)

    def _jacobian_det(self, x):
        self.BC.params.values = [self.get_nu(), self.params.values[0]]
        return self.BC.jacobian_det(x)

    def params_sample(self, nsamples=500, minval=0., maxval=1.):
        return self.BC.params_sample(nsamples, minval, maxval)[:, 1][:, None]



class BoxCox1nu(Transform):
    ''' BoxCox transform y = ((nu+x)^lambda-1)/lambda
        with lambda exponent fixed
    '''

    def __init__(self, mininu=EPS, minilam=0.):
        params = Vector(['nu'], [mininu], [mininu], [np.inf])

        # Define the nu constant and set it to inf by default
        # to force proper setup
        constants = Vector(['lam'], [np.nan], [minilam], [3], accept_nan=True)

        super(BoxCox1nu, self).__init__('BoxCox1nu', params, constants)
        self.BC = BoxCox2(mininu=mininu, minilam=minilam)

    def get_lam(self):
        lam = self.constants.values[0]
        if np.isnan(lam):
            raise ValueError('lam is nan. It must be set to a proper value')
        return lam

    def _forward(self, x):
        self.BC.params.values = [self.params.values[0], self.get_lam()]
        return self.BC.forward(x)

    def _backward(self, y):
        self.BC.params.values = [self.params.values[0], self.get_lam()]
        return self.BC.backward(y)

    def _jacobian_det(self, x):
        self.BC.params.values = [self.params.values[0], self.get_lam()]
        return self.BC.jacobian_det(x)

    def params_sample(self, nsamples=500, minval=0., maxval=1.):
        return self.BC.params_sample(nsamples, minval, maxval)[:, 0][:, None]



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

        Transform is reparameterized as (loga, logb)
    '''

    def __init__(self):
        params = Vector(['loga', 'logb'], [-1., 0.], [-20, -5.], [0., 5.])

        # Define the xmax constant and set it to nan by default
        # to force proper setup
        constants = Vector(['xmax'], [np.nan], [EPS], [np.inf], accept_nan=True)

        super(LogSinh, self).__init__('LogSinh', \
            params, constants)

    def get_xmax(self):
        xmax = self.constants.values[0]
        if np.isnan(xmax):
            raise ValueError('xmax is nan. It must be set to a proper value')
        return xmax

    def _forward(self, x):
        xmax = self.get_xmax()
        loga, logb = self.params.values
        a = math.exp(loga)
        b = math.exp(logb)

        xn = x/xmax
        w = a + b*xn
        return np.where(xn > -a/b+EPS, \
            (w+np.log((1.-np.exp(-2.*w))/2.))/b, np.nan)

    def _backward(self, y):
        xmax = self.get_xmax()
        loga, logb = self.params.values
        a = math.exp(loga)
        b = math.exp(logb)

        w = b*y
        return xmax*(y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b)

    def _jacobian_det(self, x):
        xmax = self.get_xmax()
        loga, logb = self.params.values
        a = math.exp(loga)
        b = math.exp(logb)

        xn = x/xmax
        w = a+b*xn
        return 1./xmax*np.where(xn > -a/b+EPS, 1./np.tanh(w), np.nan)

    def params_sample(self, nsamples=500, loga_scale=3., logb_sig=0.3):
        ''' Sample from informative prior '''

        # Generate parameters samples from informative prior
        loga = -np.random.exponential(scale=loga_scale, size=nsamples)
        loga = np.maximum(loga, self.params.mins[0])

        logb = np.random.normal(loc=0., scale=logb_sig, size=nsamples)
        logb = np.maximum(logb, self.params.mins[1])

        return np.column_stack([loga, logb])


    def params_logprior(self):
        ''' <0 prior for loga and N(0., 0.3) prior for logb '''
        loga, logb = self.params.values
        if loga > 0:
            return -np.inf

        return norm.logpdf(logb, loc=0., scale=0.3)


class Reciprocal(Transform):
    ''' Reciprocal transform y=-1/(nu+x) '''

    def __init__(self, mininu=EPS):
        params = Vector(['nu'], [mininu], [mininu])
        super(Reciprocal, self).__init__('Reciprocal', params)
        self.mininu = mininu

    def _forward(self, x):
        nu = self.params.values[0]
        return np.where(x > -nu, -1./(nu+x), np.nan)

    def _backward(self, y):
        nu = self.params.values[0]
        return np.where(y < -self.mininu, -1./y-nu, np.nan)

    def _jacobian_det(self, x):
        nu = self.params.values[0]
        return np.where(x > -nu, 1./(nu+x)**2, np.nan)

    def params_sample(self, nsamples=500, minval=-7., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, nsamples)
        samples = np.exp(samples)[:, None]+self.mininu

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



class Sinh(Transform):
    ''' Sinh transform y=arcsinh((x-sign(x) * log(cst+abs(x))-log(cst) '''

    def __init__(self):
        params = Vector(['nu', 'scale'], [0., 1.], [-np.inf, 1e-10], \
                                                [np.inf, np.inf])
        super(Sinh, self).__init__('Sinh', params)

    def _forward(self, x):
        nu, scale = self.params.values
        u = (x-nu)*scale
        return np.arcsinh(u)

    def _backward(self, y):
        nu, scale = self.params.values
        return np.sinh(y)/scale+nu

    def _jacobian_det(self, x):
        nu, scale = self.params.values
        u = (x-nu)*scale
        return scale/np.sqrt(1+u*u)

    def params_sample(self, nsamples=500, minval=-7., maxval=0.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, (nsamples, 2))
        samples[:, 1] = np.exp(samples[:, 1])

        return samples



class Manly(Transform):
    ''' Manly (1976) transform y=exp(lam x)-1)/lam

        Manly, B. F. J. "Exponential data transformations."
        The Statistician (1976): 37-42.
    '''

    def __init__(self):
        params = Vector(['lam'], [0.1], [-5], [5])

        # Define the xmax constant and set it to nan by default
        # to force proper setup
        constants = Vector(['xmax'], [np.nan], [EPS], [np.inf], accept_nan=True)

        super(Manly, self).__init__('Manly', params, constants)


    def get_xmax(self):
        xmax = self.constants.values[0]
        if np.isnan(xmax):
            raise ValueError('xmax is nan. It must be set to a proper value')
        return xmax


    def _forward(self, x):
        lam = self.params.values[0]
        xmax = self.get_xmax()
        if abs(lam-EPS) > 0.:
            u = x/xmax
            return (np.exp(lam*u)-1)/lam
        else:
            return u

    def _backward(self, y):
        lam = self.params.values[0]
        xmax = self.get_xmax()
        if abs(lam-EPS) > 0.:
            return xmax*np.log(1+(lam*y))/lam
        else:
            return xmax*y

    def _jacobian_det(self, x):
        lam = self.params.values[0]
        xmax = self.get_xmax()
        if abs(lam-EPS) > 0.:
            u = x/xmax
            return np.exp(lam*u)/xmax
        else:
            return np.one_likes(x)/xmax

    def params_sample(self, nsamples=500, minval=-5., maxval=5.):
        # Generate parameters samples in log space
        samples = np.random.uniform(minval, maxval, (nsamples, 1))
        return samples


