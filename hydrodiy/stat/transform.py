import math
import numpy as np

EPS = 1e-10


def inf2bound(a, amin, amax):
    ''' Convert inf2bound variable to non-bounded via logit '''
    bnd = 1-1./(1+math.exp(a))
    bnd = bnd*(amax-amin) + amin
    return bnd

def bound2inf(bnd, amin, amax):
    ''' Convert non-bounded variable to inf2bound via logit '''
    value = min(1-EPS, max(EPS, (bnd-amin)/(amax-amin)))
    if value < EPS:
        return amin
    elif value > 1.-EPS:
        return amax
    else:
        value = math.log(1./(1-value)-1)
    return value


def get_transform(name):
    ''' Returns an instance of a particular transform '''

    if name == 'Identity':
        return IdentityTransform

    if name == 'Bound':
        return BoundTransform

    if name == 'Log':
        return LogTransform

    elif name == 'BoxCox':
        return BoxCoxTransform

    elif name == 'YeoJohnson':
        return YeoJTransform

    elif name == 'LogSinh':
        return LogSinhTransform

    else:
        raise ValueError('Cannot recognised transform ' + name)

    return trans



class Transform(object):
    ''' Simple interface to common transform functions '''

    def __init__(self, name, ntparams,
            nrparams=None,  \
            nconstants=0, \
            rparams_mins=None, \
            rparams_maxs=None, \
            constants=None):
        ''' Initialise transform object with number of transformed
        parameters (ntparams) and _name. Number of raw parameters
        is optional, will be set to ntparams by default.
        Number of constants is set to 0 by default.
        '''
        self._name = name

        # Value added to avoid nan around 0.
        self._eps = EPS

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
        if nconstants == 0:
            self._constants = None
        else:
            constants = np.atleast_1d(constants).astype(float)

            if constants.shape[0] != nconstants:
                raise ValueError(('Expected constants of length' + \
                    ' {0}, got {1}').format(nconstants, \
                        constants.shape[0]))

            self._constants = constants


    def __str__(self):
        s = '\n{0} transform\n'.format(self._name)
        if not self._constants is None :
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

    def __init__(self, constants=None):
        Transform.__init__(self, 'Identity', \
                ntparams=0, \
                nconstants=0)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def jacobian_det(self, x):
        return np.ones_like(x)



class BoundTransform(Transform):

    def __init__(self, constants=[-10., 10.]):

        Transform.__init__(self, 'Bound', \
                ntparams=0, \
                nconstants=2, \
                constants=constants)

    def forward(self, x):
        lower, upper = self._constants
        value = (x-lower)/(upper-lower)
        y = np.log(1./(1-value)-1)
        return y

    def backward(self, y):
        lower, upper = self._constants
        bnd = 1-1./(1+np.exp(y))
        x = bnd*(upper-lower) + lower
        return x

    def jacobian_det(self, x):
        lower, upper = self._constants
        value = (x-lower)/(upper-lower)
        j = 1./(upper-lower) / value / (1-value)
        j[value<EPS] = np.nan
        j[value>1-EPS] = np.nan
        return j



class LogTransform(Transform):

    def __init__(self, constants=1e-3):

        Transform.__init__(self, 'Log', \
                ntparams=0, \
                nconstants=1, \
                constants=constants)

    def forward(self, x):
        cst = self._constants[0]
        y = np.log(x+cst)
        y[x<0] = np.nan
        return y

    def backward(self, y):
        cst = self._constants[0]
        x =  np.exp(y)-cst
        return x

    def jacobian_det(self, x):
        cst = self._constants[0]
        j = 1./(x+cst)
        j[x<0] = np.nan
        return j



class BoxCoxTransform(Transform):

    def __init__(self, constants=1e-3):

        Transform.__init__(self, 'BoxCox',
            ntparams=1,
            nconstants=1, \
            constants=constants, \
            rparams_mins=1e-5, \
            rparams_maxs=3.)

    def _trans2raw(self):
        self._rparams[0] = inf2bound(self._tparams[0], 1e-2, 3.)

    def _raw2trans(self):
        self._tparams[0] = bound2inf(self._rparams[0], 1e-2, 3.)

    def forward(self, x):
        cst = self._constants[0]
        lam = self._rparams[0]
        y = (np.power(x+cst, lam)-cst**lam)/lam
        return y

    def backward(self, y):
        cst = self._constants[0]
        lam = self._rparams[0]
        x =  np.power(lam*y+cst**lam, 1./lam)-cst
        return x

    def jacobian_det(self, x):
        cst = self._constants[0]
        lam = self._rparams[0]
        j = np.power(x+cst,lam-1.)
        return j



class YeoJTransform(Transform):

    def __init__(self, constants=None):

        Transform.__init__(self, 'YeoJohnson',
            ntparams=3, \
            nconstants=0, \
            constants=constants, \
            rparams_mins=[-np.inf, 0., -1.], \
            rparams_maxs=[np.inf, np.inf, 3.])

    def _trans2raw(self):
        self._rparams[0] = self._tparams[0]
        self._rparams[1] = math.exp(self._tparams[1])
        self._rparams[2] = inf2bound(self._tparams[2], -1., 3.)

    def _raw2trans(self):
        self._tparams[0] = self._rparams[0]
        self._tparams[1] = math.log(self._rparams[1])
        self._tparams[2] = bound2inf(self._rparams[2], -1., 3.)

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

    def __init__(self, constants=10.):

        Transform.__init__(self, 'LogSinh', \
            ntparams=1, \
            nconstants=1, \
            constants=constants, \
            rparams_mins=1e-5, \
            rparams_maxs=0.93)

    def _trans2raw(self):
        self._rparams[0] = inf2bound(self._tparams[0], 0.3, 0.93)

    def _raw2trans(self):
        self._tparams[0] = bound2inf(self._rparams[0], 0.3, 0.93)

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


