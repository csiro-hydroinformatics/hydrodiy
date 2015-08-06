import numpy as np


class Transform:
    ''' Simple interface to common transform functions '''

    def __init__(self, nparams, name):
        self.name = name
        self.nparams = nparams
        self.params = np.array([np.nan] * nparams)

    def __str__(self):
        s = '\n%s transform\n params = [' % self.name
        for i in range(self.nparams):
            s += '%0.2f, ' % self.params[i]
        s = s[:-2] + ']'

        return s

    def paramcheck(self):
        ''' Check parameter transforms '''
        pass

    def forward(self, x):
        ''' Returns the forward transform of x '''
        return np.nan

    def inverse(self, y):
        ''' Returns the inverse transform of x '''
        return np.nan

    def jac(self, x):
        ''' Returns the transformation jacobian dforward(x)/dx '''
        return np.nan


class LogTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 1, 'Log')

    def forward(self, x):
        b = math.exp(self.params[0])
        f = 1/b * np.log(1.+b*x)
        return f

    def inverse(self, y):
        b = math.exp(self.params[0])
        i =  (np.exp(b*y)-1.)/b  
        return i

    def jac(self, x):
        b = math.exp(self.params[0])
        j =  1./(1.+b*x) 
        return j


class YeoJohsonTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 2, 'Yeo-Johnson')

    def forward(self, x):
        # TODO
        b = math.exp(self.params[0])
        f = 1/b * np.log(1.+b*x)
        return f

    def inverse(self, y):
        # TODO
        b = math.exp(self.params[0])
        i =  (np.exp(b*y)-1.)/b  
        return i

    def jac(self, x):
        # TODO
        b = math.exp(self.params[0])
        j =  1./(1.+b*x) 
        return j

    #def _parcheck(self):
    #    ''' Check parameter bounds '''
    #    if self.scale<=0:
    #        raise ValueError('scale parameter <0')

    #    if self._name == 'yeojohnson':
    #        if (self.shape<0)|(self.shape>5):
    #            raise ValueError('Yeo-Jonhson shape parameter <0 or >5')

    #def getname(self):
    #    return self._name

    #def r2t(self, x):
    #    ''' Transform raw inputut data '''
    #    self._parcheck()
    #    input = self._toarray(x)
    #    if self._name == 'yeojohnson':
    #        output = self._r2t_yj(input)
    #    if self._name == 'logsinh':
    #        output = self._r2t_logsinh(input)
    #    
    #    return output

    #def t2r(self, x):
    #    ''' Inverse transform function '''
    #    self._parcheck()
    #    input = self._toarray(x)
    #    if self._name == 'yeojohnson':
    #        output = self._t2r_yj(input)
    #    if self._name == 'logsinh':
    #        output = self._t2r_logsinh(input)
    #    
    #    return output

    #def jac(self, x):
    #    ''' Jacobian of transform function '''
    #    self._parcheck()
    #    input = self._toarray(x)
    #    if self._name == 'yeojohnson':
    #        output = self._jac_yj(input)
    #    if self._name == 'logsinh':
    #        output = self._jac_logsinh(input)

    #    return output

    #def _toarray(self, input):
    #    output = np.array(input)
    #    if len(output.shape)==0:
    #        output = output.reshape((1,1))
    #    return output

    #def _r2t_yj(self, input):
    #    ''' Compute the yeojohnson transform with a single parameter '''
    #    output = input*np.nan
    #    w = self.loc+input*self.scale
    #    expon = self.shape
    #    ipos = w >=0
    #    if not np.isclose(expon, 0.0):
    #        output[ipos] = ((w[ipos]+1)**expon-1)/expon
    #    if np.isclose(expon, 0.0):
    #        output[ipos] = np.log(w[ipos]+1)
    #    if not np.isclose(expon, 2.0):
    #        output[~ipos] = -((-w[~ipos]+1)**(2-expon)-1)/(2-expon)
    #    if np.isclose(expon, 2.0):
    #        output[~ipos] = -np.log(-w[~ipos]+1)

    #    return output

    #def _t2r_yj(self, input):
    #    ''' Compute the yeojohnson transform 
    #        with a single parameter 
    #    '''
    #    output = input*np.nan
    #    expon = self.shape
    #    ipos = input >=0
    #    if not np.isclose(expon, 0.0):
    #        output[ipos] = (expon*input[ipos]+1)**(1/expon)-1
    #    if np.isclose(expon, 0.0):
    #        output[ipos] = np.exp(input[ipos])-1
    #    if not np.isclose(expon, 2.0):
    #        output[~ipos] = -(-(2-expon)*input[~ipos]+1)**(1/(2-expon))+1
    #    if np.isclose(expon, 2.0):
    #        output[~ipos] = -np.exp(-input[~ipos])+1

    #    return (output-self.loc)/self.scale
    #
    #def _jac_yj(self, input):
    #    ''' Compute the jac of the yeojohnson transform 
    #        with a single parameter 
    #    '''
    #    output = input*np.nan
    #    expon = self.shape
    #    w = self.loc+input*self.scale
    #    ipos = w >=0
    #
    #    if not np.isclose(expon, 0.0):
    #        output[ipos] = (w[ipos]+1)**(expon-1)
    #    if np.isclose(expon, 0.0):
    #        output[ipos] = 1/(w[ipos]+1)
    #    if not np.isclose(expon, 2.0):
    #        output[~ipos] = (-w[~ipos]+1)**(1-expon)
    #    if np.isclose(expon, 2.0):
    #        output[~ipos] = 1/(-w[~ipos]+1)
    #
    #    return output*self.scale


class LogSinhTransform(Transform):

    def __init__(self):
        Transform.__init__(self, 2, 'LogSinh')

    def forward(self, x):
        # TODO
        b = math.exp(self.params[0])
        f = 1/b * np.log(1.+b*x)
        return f

    def inverse(self, y):
        # TODO
        b = math.exp(self.params[0])
        i =  (np.exp(b*y)-1.)/b  
        return i

    def jac(self, x):
        # TODO
        b = math.exp(self.params[0])
        j =  1./(1.+b*x) 
        return j

    #def _r2t_logsinh(self, input):
    #    ''' Compute the log-sinh transform '''
    #    a = self.loc
    #    b = self.scale
    #    w = a + b*input
    #    output = input*np.nan
    #    idx = w<10.

    #    if np.sum(idx)>0:
    #        output[idx] = np.log(np.sinh(w[idx]))/b
    #    if np.sum(~idx)>0:
    #        output[~idx] = (w[~idx]+np.log((1.-np.exp(-2.*w[~idx]))/2.))/b 
    #                
    #    return output
    #
    #def _t2r_logsinh(self, input):
    #    ''' Compute the inverse log-sinh transform '''
    #    a = self.loc
    #    b = self.scale
    #    w = b*input
    #    output = input*np.nan
    #    idx = w<10.

    #    if np.sum(idx)>0:
    #        output[idx] = (np.arcsinh(np.exp(w[idx])) - a)/b
    #    if np.sum(~idx)>0:
    #        output[~idx] = input[~idx]
    #        output[~idx] += (np.log(1.+np.sqrt(1.+np.exp(-2.*w[~idx])))-a)/b 
    #    return output

    #def _jac_logsinh(self, input): 
    #    ''' Jacobian of log-sinh transform '''
    #    a = self.loc
    #    b = self.scale
    #    w = a + b*input
    #    return 1./np.tanh(w)



