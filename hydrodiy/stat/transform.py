import math
import numpy as np


def forward_bounded(a, amin, amax):
    ''' Convert forward_bounded variable to non-bounded via logit '''
    bnd = 1-1./(1+math.exp(a))
    bnd = bnd*(amax-amin) + amin
    return bnd

def backward_bounded(bnd, amin, amax, eps=1e-10):
    ''' Convert non-bounded variable to forward_bounded via logit '''
    value = (bnd-amin)/(amax-amin)
    if value < eps:
        return amin
    elif value > 1.-eps:
        return amax
    else:
        value = math.log(1./(1-value)-1)
    return value


def getinstance(_name):
    ''' Returns an instance of a particular transform '''

    if _name == IdentityTransform()._name:
        return IdentityTransform()

    if _name == LogTransform()._name:
        return LogTransform()

    elif _name == BoxCoxTransform()._name:
        return BoxCoxTransform()

    elif _name == YeoJTransform()._name:
        return YeoJTransform()

    elif _name == LogSinhTransform()._name:
        return LogSinhTransform()

    else:
        raise ValueError('Cannot find transform _name %s' % _name)

    return trans



class Transform(object):
    ''' Simple interface to common transform functions '''

    def __init__(self, name, ntparams,
            nrparams=None,  \
            rparams_mins=None, \
            rparams_maxs=None, \
            nconstants=0):
        ''' Initialise transform object with number of transformed
        parameters (ntparams) and _name. Number of raw parameters
        is optional, will be set to ntparams by default.
        Number of constants is set to 0 by default.
        '''
        self._name = name

        # Value added to avoid nan around 0.
        self._eps = 1e-10

        # Initialise trans params
        self._ntparams = ntparams
        if ntparams > 0:
            self._tparams = np.nan * np.ones(ntparams)
        else:
            self._tparams = None

        # Initialise raw params
        if nrparams is None:
            nrparams = ntparams

        self._nrparams = nrparams
        self._rparams = np.nan * np.ones(nrparams)

        if rparams_mins is None:
            rparams_mins = [-np.inf] * nrparams

        self._rparams_mins = np.atleast_1d(rparams_mins)
        if len(self._rparams_mins) != nrparams:
            raise ValueError('Wrong length of mins')

        if rparams_maxs is None:
            rparams_maxs = [np.inf] * nrparams

        self._rparams_maxs = np.atleast_1d(rparams_maxs)
        if len(self._rparams_maxs) != nrparams:
            raise ValueError('Wrong length of maxs')

        # Initialise constants
        self._nconstants = nconstants
        if nrparams > 0:
            self._constants = np.nan * np.ones(nconstants)
        else:
            self._constants = None


    def __str__(self):
        s = '\n{0} transform\n'.format(self._name)
        if self._nconstants > 0:
            s += '  Constants = [' + ', '.join(['{0:3.3e}'.format(cst) \
                for cst in  self._constants]) + ']'
        if self._nrparams > 0:
            s += '  Raw params = [' + ', '.join(['{0:3.3e}'.format(rp) \
                for rp in  self._rparams]) + ']'
        if self._ntparams > 0:
            s += '  Trans params = [' + ', '.join(['{0:3.3e}'.format(rt) \
                for rt in  self._tparams]) + ']'
        s += '\n'
        return s


    @property
    def name(self):
        return self._name


    @property
    def ntparams(self):
        return self._ntparams


    @property
    def nrparams(self):
        return self._nrparams


    @property
    def nconstants(self):
        return self._nconstants


    @property
    def rparams(self):
        return self._rparams

    @rparams.setter
    def rparams(self, value):
        if self.nrparams == 0:
            return

        value = np.atleast_1d(value).astype(np.float64)
        if len(value) != self._nrparams:
            raise ValueError('Wrong number of raw parameters' + \
                ' ({0}), should be {1}'.format(len(value),
                    self._nrparams))
        self._rparams = np.clip(value, self._rparams_mins, self._rparams_maxs)
        self._raw2trans()


    @property
    def constants(self):
        return self._constants

    @constants.setter
    def constants(self, value):
        if self.nconstants == 0:
            return

        value = np.atleast_1d(value).astype(np.float64)
        if len(value) != self._nconstants:
            raise ValueError('Wrong number of constants' + \
                ' ({0}), should be {1}'.format(len(value),
                    self._nconstants))
        self._constants = value


    @property
    def tparams(self):
        return self._tparams

    @tparams.setter
    def tparams(self, value):
        if self.ntparams == 0:
            return

        value = np.atleast_1d(value).astype(np.float64)
        if len(value) != self._ntparams:
            raise ValueError('Wrong number of trans parameters' + \
                ' ({0}), should be {1}'.format(len(value),
                    self._ntparams))
        self._tparams = value
        self._trans2raw()


    def _trans2raw(self):
        ''' Converts transformed parameters into raw parameters
            Does nothing by default.
        '''
        self._rparams = self._tparams.copy()

    def _raw2trans(self):
        ''' Converts raw parameters into transformed parameters
            Does nothing by default.
        '''
        self._tparams = self._rparams.copy()


    def forward(self, x):
        ''' Returns the forward transform of x '''
        raise NotImplementedError('Method forward not implemented')

    def backward(self, y):
        ''' Returns the backward transform of x '''
        raise NotImplementedError('Method backward not implemented')

    def jacobian_det(self, x):
        ''' Returns the transformation jacobian_detobian d[forward(x)]/dx '''
        raise NotImplementedError('Method jacobian_det not implemented')


class IdentityTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 'Identity', 0)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def jacobian_det(self, x):
        return np.ones_like(x)



class LogTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 'Log', 0,
            nconstants=1)
        self.constants = 0.

    def forward(self, x):
        cst = self._constants[0]
        y = np.where(x>0, np.log(x+cst), np.nan)
        return y

    def backward(self, y):
        cst = self._constants[0]
        x =  np.where(y>math.log(cst), np.exp(y)-cst, np.nan)
        return x

    def jacobian_det(self, x):
        cst = self._constants[0]
        j = np.where(x>0, 1./(x+cst), np.nan)
        return j



class BoxCoxTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 'BoxCox', 1,
            rparams_mins=1e-2,
            rparams_maxs=3.)

    def _trans2raw(self):
        self._rparams[0] = forward_bounded(self._tparams[0], 1e-2, 3.)

    def _raw2trans(self):
        self._tparams[0] = backward_bounded(self._rparams[0], 1e-2, 3.)

    def forward(self, x):
        lam = self._rparams[0]
        y = np.where(x>0, (np.power(x,lam)-1.)/lam, np.nan)
        return y

    def backward(self, y):
        lam = self._rparams[0]
        x =  np.where(y>-1./lam, np.power(lam*y+1., 1./lam), np.nan)
        return x

    def jacobian_det(self, x):
        lam = self._rparams[0]
        j = np.where(x>0, np.power(x,lam-1.), np.nan)
        return j



class YeoJTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 'YeoJohnson', 3,
            rparams_mins=[-np.inf, 0., -1.],
            rparams_maxs=[np.inf, np.inf, 3.])

    def _trans2raw(self):
        self._rparams[0] = self._tparams[0]
        self._rparams[1] = math.exp(self._tparams[1])
        self._rparams[2] = forward_bounded(self._tparams[2], -1., 3.)

    def _raw2trans(self):
        self._tparams[0] = self._rparams[0]
        self._tparams[1] = math.log(self._rparams[1])
        self._tparams[2] = backward_bounded(self._rparams[2], -1., 3.)

    def forward(self, x):
        loc, scale, expon = self._rparams
        y = x*np.nan
        w = loc+x*scale
        ipos = w >= 0

        if not np.isclose(expon, 0.0):
            y[ipos] = (np.power(w[ipos]+1, expon)-1)/expon

        if np.isclose(expon, 0.0):
            y[ipos] = np.log(w[ipos]+1)

        if not np.isclose(expon, 2.0):
            y[~ipos] = -(np.power(-w[~ipos]+1, 2-expon)-1)/(2-expon)

        if np.isclose(expon, 2.0):
            y[~ipos] = -np.log(-w[~ipos]+1)

        return y


    def backward(self, y):
        loc, scale, expon = self._rparams
        x = y*np.nan
        ipos = y >= 0

        if not np.isclose(expon, 0.0):
            x[ipos] = np.power(expon*y[ipos]+1, 1./expon)-1
        else:
            x[ipos] = np.exp(y[ipos])-1

        if not np.isclose(expon, 2.0):
            x[~ipos] = -np.power(-(2-expon)*y[~ipos]+1, 1./(2-expon))+1
        else:
            x[~ipos] = -np.exp(-y[~ipos])+1

        return (x-loc)/scale


    def jacobian_det(self, x):
        loc, scale, expon = self._rparams
        j = x*np.nan
        w = loc+x*scale
        ipos = w >=0

        if not np.isclose(expon, 0.0):
            j[ipos] = (w[ipos]+1)**(expon-1)

        if np.isclose(expon, 0.0):
            j[ipos] = 1/(w[ipos]+1)

        if not np.isclose(expon, 2.0):
            j[~ipos] = (-w[~ipos]+1)**(1-expon)

        if np.isclose(expon, 2.0):
            j[~ipos] = 1/(-w[~ipos]+1)

        return j*scale


def logsinh_ab(eps):
    A = math.sqrt(4788.*eps*eps-8467.2*eps+3969.)
    nu = math.sqrt((210.*eps-157.5+2.5*A)/(14.-15.*eps))

    c_constant = 1e-2
    a = nu/(1+5./c_constant)
    b = a/c_constant

    return a, b


class LogSinhTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 'LogSinh', 1, nconstants=1,
            rparams_mins=0.3,
            rparams_maxs=0.93)

    def _trans2raw(self):
        self._rparams[0] = forward_bounded(self._tparams[0], 0.3, 0.93)


    def _raw2trans(self):
        self._tparams[0] = backward_bounded(self._rparams[0], 0.3, 0.93)

    def forward(self, x):
        a, b = logsinh_ab(self._rparams[0])
        xmax = self._constants[0]
        u = 5.*x/xmax
        w = a + b*u
        y = np.where(x>0, (w+np.log((1.-np.exp(-2.*w))/2.))/b, np.nan)

        return y

    def backward(self, y):
        a, b = logsinh_ab(self._rparams[0])
        xmax = self._constants[0]
        w = b*y
        output = y*np.nan
        u = y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b
        x = xmax*u/5

        return x


    def jacobian_det(self, x):
        a, b = logsinh_ab(self._rparams[0])
        xmax = self._constants[0]
        u = 5.*x/xmax
        w = a + b*u
        jac = np.where(x>0, 5./xmax/np.tanh(w), np.nan)

        return jac


