import math
import numpy as np


def bounded(a, amin, amax):
    ab = 1-1./(1+math.exp(a))
    ab = ab*(amax-amin) + amin
    return ab

def inversebounded(ab, amin, amax, eps=1e-10):
    b = (ab-amin)/(amax-amin)
    if b < eps:
        return amin
    elif b > 1.-eps:
        return amax
    else:
        b = math.log(1./(1-b)-1)
    return b


def getinstance(name):
    ''' Returns an instance of a particular transform '''

    if name == LogTrans().name:
        return LogTrans()

    elif name == PowerTrans().name:
        return PowerTrans()

    elif name == YeoJohnsonTrans().name:
        return YeoJohnsonTrans()

    elif name == LogSinhTrans().name:
        return LogSinhTrans()

    else:
        raise ValueError('Cannot find transform name %s' % name)

    return trans



class Transform:
    ''' Simple interface to common transform functions '''

    def __init__(self, nparams, name):
        self.name = name
        self.nparams = nparams
        self.params = np.array([np.nan] * nparams)

    def __str__(self):
        s = '\n%s transform. Params = [' % self.name

        tp = self.trueparams()
        if not isinstance(tp, list):
            tp = [tp]

        for i in range(self.nparams):
            s += '%0.2f, ' % tp[i]
        s = s[:-2] + ']'

        return s

    def trueparams(self):
        ''' Returns true parameter values '''
        return [np.nan] * self.nparams

    def forward(self, x):
        ''' Returns the forward transform of x '''
        return np.nan

    def inverse(self, y):
        ''' Returns the inverse transform of x '''
        return np.nan

    def jac(self, x):
        ''' Returns the transformation jacobian dforward(x)/dx '''
        return np.nan



class IdentityTrans(Transform):

    def __init__(self):
        Transform.__init__(self, 0, 'Identity')

    def trueparams(self):
        return np.nan

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def jac(self, x):
        return np.one_like(x)



class LogTrans(Transform):

    def __init__(self):
        Transform.__init__(self, 1, 'Log')

    def trueparams(self):
        return math.exp(self.params[0])

    def forward(self, x):
        c = self.trueparams()
        y = np.log(c+x)
        return y

    def inverse(self, y):
        c = self.trueparams()
        x =  np.exp(y)-c
        return x

    def jac(self, x):
        c = self.trueparams()
        j = np.nan * x
        idx = c+x > 0.
        j[idx] =  1./(c+x[idx])
        return j



class PowerTrans(Transform):

    def __init__(self):
        Transform.__init__(self, 1, 'Power')

    def trueparams(self):
        return bounded(self.params[0], 0, 4)

    def forward(self, x):
        b = self.trueparams()
        y = ((1.+x)**b-1.)/b
        return y

    def inverse(self, y):
        b = self.trueparams()
        x =  (b*y+1.)**(1/b)-1.
        # Highly unreliable for b<0 and if y -> -1/b
        return x

    def jac(self, x):
        b = self.trueparams()
        j = (1.+x)**(b-1)
        return j



class YeoJohnsonTrans(Transform):

    def __init__(self):
        Transform.__init__(self, 3, 'YeoJohnson')

    def trueparams(self):
        return self.params[0], \
                math.exp(self.params[1]), \
                bounded(self.params[2], -5, 5)

    def forward(self, x):

        loc, scale, expon = self.trueparams()

        y = x*np.nan
        w = loc+x*scale

        ipos = w >= 0

        if not np.isclose(expon, 0.0):
            y[ipos] = ((w[ipos]+1)**expon-1)/expon

        if np.isclose(expon, 0.0):
            y[ipos] = np.log(w[ipos]+1)

        if not np.isclose(expon, 2.0):
            y[~ipos] = -((-w[~ipos]+1)**(2-expon)-1)/(2-expon)

        if np.isclose(expon, 2.0):
            y[~ipos] = -np.log(-w[~ipos]+1)

        return y


    def inverse(self, y):

        loc, scale, expon = self.trueparams()

        x = y*np.nan

        ipos = x >=0

        if not np.isclose(expon, 0.0):
            x[ipos] = (expon*y[ipos]+1)**(1/expon)-1
        else:
            x[ipos] = np.exp(y[ipos])-1

        if not np.isclose(expon, 2.0):
            x[~ipos] = -(-(2-expon)*y[~ipos]+1)**(1/(2-expon))+1
        else:
            x[~ipos] = -np.exp(-y[~ipos])+1

        return (x-loc)/scale


    def jac(self, x):

        loc, scale, expon = self.trueparams()

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



class LogSinhTrans(Transform):

    def __init__(self):
        Transform.__init__(self, 2, 'LogSinh')

    def trueparams(self):
        return math.exp(self.params[0]), \
                math.exp(self.params[1])

    def forward(self, x):

        a, b = self.trueparams()
        w = a + b*x
        y = x*np.nan
        y = (w+np.log((1.-np.exp(-2.*w))/2.))/b

        return y


    def inverse(self, y):

        a, b = self.trueparams()
        w = b*y
        output = y*np.nan
        x = y + (np.log(1.+np.sqrt(1.+np.exp(-2.*w)))-a)/b

        return x


    def jac(self, x):
        a, b = self.trueparams()
        w = a + b*x

        return 1./np.tanh(w)



