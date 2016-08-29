import math
import numpy as np


def bounded(a, amin, amax):
    ''' Convert bounded variable to non-bounded via logit '''
    bnd = 1-1./(1+math.exp(a))
    bnd = bnd*(amax-amin) + amin
    return bnd

def backward_bounded(bnd, amin, amax, eps=1e-10):
    ''' Convert non-bounded variable to bounded via logit '''
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

    def __init__(self, ntparams, name, nrparams=None):
        ''' Initialise transform object with number of transformed
        parameters (ntparams) and _name. Number of raw parameters
        is optional, will be set to ntparams by default.
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
        if nrparams > 0:
            self._rparams = np.nan * np.ones(nrparams)
        else:
            self._rparams = None


    def __str__(self):
        s = '\n{0} transform\n'.format(self._name)
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
        value = np.atleast_1d(value).astype(np.float64)
        if len(value) != self._nrparams:
            raise ValueError('Wrong number of raw parameters' + \
                ' ({0}), should be {1}'.format(len(value),
                    self._nrparams))
        self._rparams = value
        self._raw2trans()


    @property
    def tparams(self):
        return self._tparams

    @tparams.setter
    def tparams(self, value):
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
        Transform.__init__(self, 0, 'Identity')

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def jacobian_det(self, x):
        return np.ones_like(x)



class LogTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 1, 'Log')

    def _trans2raw(self):
        self._rparams[0] = math.exp(self._tparams[0])

    def _raw2trans(self):
        self._tparams[0] = math.log(self._rparams[0])

    def forward(self, x):
        cst = self._rparams[0]
        x = np.clip(x, -cst+self._eps, np.inf)
        y = np.log(cst+x)
        return y

    def backward(self, y):
        cst = self._rparams[0]
        x =  np.exp(y)-cst
        return x

    def jacobian_det(self, x):
        cst = self._rparams[0]
        x = np.clip(x, -cst+self._eps, np.inf)
        j = np.nan * x
        idx = cst+x > 0.
        j[idx] =  1./(cst+x[idx])
        return j



class BoxCoxTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 1, 'BoxCox')

    def _trans2raw(self):
        self._rparams[0] = bounded(self._tparams[0], 0, 4)

    def _raw2trans(self):
        self._tparams[0] = backward_bounded(self._rparams[0], 0, 4)

    def forward(self, x):
        lam = self._rparams[0]
        y = np.sign(x)*(np.power(1.+np.abs(x),lam)-1.)/lam
        return y

    def backward(self, y):
        lam = self._rparams[0]
        x =  np.sign(y)*np.power(lam*np.abs(y)+1., 1./lam)-1.
        # Highly unreliable for b<0 and if y -> -1/b
        return x

    def jacobian_det(self, x):
        lam = self._rparams[0]
        j = np.sign(x)*np.power(1.+np.abs(x), lam-1)
        return j



class YeoJTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 3, 'YeoJohnson')

    def _trans2raw(self):
        self._rparams[0] =  self._tparams[0]
        self._rparams[1] =  math.exp(self._tparams[1])
        self._rparams[2] =  bounded(self._tparams[2], -1, 3)

    def _raw2trans(self):
        self._tparams[0] =  self._rparams[0]
        self._tparams[1] =  math.log(self._rparams[1])
        self._tparams[2] =  backward_bounded(self._rparams[2], -1, 3)

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



class LogSinhTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 2, 'LogSinh')

    def _trans2raw(self):
        self._rparams = np.exp(self._tparams)

    def _raw2trans(self):
        self._rparams = np.log(self._rparams)

    def forward(self, x):
        a, b = self._rparams
        x = np.clip(x, -a/b+self._eps, np.inf)
        w = a + b*x
        y = x*np.nan
        y = (w+np.log((1.-np.exp(-2.*w))/2.))/b

        return y


    def backward(self, y):
        a, b = self._rparams
        w = b*y
        output = y*np.nan
        x = y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b

        return x


    def jacobian_det(self, x):
        a, b = self._rparams
        x = np.clip(x, -a/b+self._eps, np.inf)
        w = a + b*x

        return 1./np.tanh(w)



