import numpy as np

class Transform:
    ''' Simple interface to common transform functions '''

    def __init__(self, name):
        if not name in ['yeojohnson', 'logsinh']:
            raise ValueError('Only "yeojonhson" and "logsinh" transform allowed')
        self._name = name
        self.loc = 0.
        self.scale = 1.
        self.shape = 1.

    def __str__(self):
        str = 'Interface to the %s transform with\n'%self._name
        str += '    location param. = %f\n'%self.loc
        str += '    scale param. = %f\n'%self.scale
        str += '    shape param. = %f\n'%self.shape
        return str

    def _parcheck(self):
        ''' Check parameter bounds '''
        if self.scale<=0:
            raise ValueError('scale parameter <0')

        if self._name == 'yeojohnson':
            if (self.shape<0)|(self.shape>5):
                raise ValueError('Yeo-Jonhson shape parameter <0 or >5')

    def getname(self):
        return self._name

    def r2t(self, x):
        ''' Transform raw inputut data '''
        self._parcheck()
        input = self._toarray(x)
        if self._name == 'yeojohnson':
            output = self._r2t_yj(input)
        if self._name == 'logsinh':
            output = self._r2t_logsinh(input)
        
        return output

    def t2r(self, x):
        ''' Inverse transform function '''
        self._parcheck()
        input = self._toarray(x)
        if self._name == 'yeojohnson':
            output = self._t2r_yj(input)
        if self._name == 'logsinh':
            output = self._t2r_logsinh(input)
        
        return output

    def jac(self, x):
        ''' Jacobian of transform function '''
        self._parcheck()
        input = self._toarray(x)
        if self._name == 'yeojohnson':
            output = self._jac_yj(input)
        if self._name == 'logsinh':
            output = self._jac_logsinh(input)

        return output

    def _toarray(self, input):
        output = np.array(input)
        if len(output.shape)==0:
            output = output.reshape((1,1))
        return output

    def _r2t_yj(self, input):
        ''' Compute the yeojohnson transform with a single parameter '''
        output = input*np.nan
        w = self.loc+input*self.scale
        expon = self.shape
        ipos = w >=0
        if not np.isclose(expon, 0.0):
            output[ipos] = ((w[ipos]+1)**expon-1)/expon
        if np.isclose(expon, 0.0):
            output[ipos] = np.log(w[ipos]+1)
        if not np.isclose(expon, 2.0):
            output[~ipos] = -((-w[~ipos]+1)**(2-expon)-1)/(2-expon)
        if np.isclose(expon, 2.0):
            output[~ipos] = -np.log(-w[~ipos]+1)

        return output

    def _t2r_yj(self, input):
        ''' Compute the yeojohnson transform 
            with a single parameter 
        '''
        output = input*np.nan
        expon = self.shape
        ipos = input >=0
        if not np.isclose(expon, 0.0):
            output[ipos] = (expon*input[ipos]+1)**(1/expon)-1
        if np.isclose(expon, 0.0):
            output[ipos] = np.exp(input[ipos])-1
        if not np.isclose(expon, 2.0):
            output[~ipos] = -(-(2-expon)*input[~ipos]+1)**(1/(2-expon))+1
        if np.isclose(expon, 2.0):
            output[~ipos] = -np.exp(-input[~ipos])+1

        return (output-self.loc)/self.scale
    
    def _jac_yj(self, input):
        ''' Compute the jac of the yeojohnson transform 
            with a single parameter 
        '''
        output = input*np.nan
        expon = self.shape
        w = self.loc+input*self.scale
        ipos = w >=0
    
        if not np.isclose(expon, 0.0):
            output[ipos] = (w[ipos]+1)**(expon-1)
        if np.isclose(expon, 0.0):
            output[ipos] = 1/(w[ipos]+1)
        if not np.isclose(expon, 2.0):
            output[~ipos] = (-w[~ipos]+1)**(1-expon)
        if np.isclose(expon, 2.0):
            output[~ipos] = 1/(-w[~ipos]+1)
    
        return output*self.scale

    def _r2t_logsinh(self, input):
        ''' Compute the log-sinh transform '''
        a = self.loc
        b = self.scale
        w = a + b*input
        output = input*np.nan
        idx = w<10.

        if np.sum(idx)>0:
            output[idx] = np.log(np.sinh(w[idx]))/b
        if np.sum(~idx)>0:
            output[~idx] = (w[~idx]+np.log((1.-np.exp(-2.*w[~idx]))/2.))/b 
                    
        return output
    
    def _t2r_logsinh(self, input):
        ''' Compute the inverse log-sinh transform '''
        a = self.loc
        b = self.scale
        w = b*input
        output = input*np.nan
        idx = w<10.

        if np.sum(idx)>0:
            output[idx] = (np.arcsinh(np.exp(w[idx])) - a)/b
        if np.sum(~idx)>0:
            output[~idx] = input[~idx]
            output[~idx] += (np.log(1.+np.sqrt(1.+np.exp(-2.*w[~idx])))-a)/b 
        return output

    def _jac_logsinh(self, input): 
        ''' Jacobian of log-sinh transform '''
        a = self.loc
        b = self.scale
        w = a + b*input
        return 1./np.tanh(w)

