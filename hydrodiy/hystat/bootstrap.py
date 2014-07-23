import math
import numpy as np
import pandas as pd
from scipy.stats import norm

class Bootstrap:
    ''' Class to perform bootstrap analysis '''

    def __init__(self, data, 
            statistic=lambda u: [np.mean(u), np.std(u)], 
            nboot=1000):
        ''' Initialise bootstrap tool

            :param np.array data: input data. Resampling is performed
                by random indexing on first dimension of data. 
                Eg: 
                    kk = np.random.choice(range(len(data)), len(data))
                    data[kk]
            :param object statistic: Function returning
                estimates from data. Must return an object
                that can be cast to numpy array (e.g. np.mean)
            :param int nboot: Number of bootstrap replicates
        '''
        # build bootstrap tool 
        self._data = np.array(data)
        if len(self._data.shape)>2:
            raise ValueError('Cannot manage data with dimensions>2')

        self.ndatapts = self._data.shape[0]
        self._nboot = nboot
        self._statistic = statistic
        
        # Get estimate and convert to vector
        self.estimate = np.array(statistic(data))

        if len(self.estimate.shape)==0:
            self.estimate = self.estimate.reshape((1, 1))

        if len(self.estimate.shape)==2:
            if self.estimate.shape[1]>1:
                mess = 'Cannot manage estimates with dimensions>1'
                raise ValueError(mess)

            nv = self.estimate.shape[0]
            self.estimate = self.estimate.reshape((nv, 1))

        self.nvar = self.estimate.shape[0]

        # Run jacknife
        self.jacknife = self._get_jacknife()

        # Run bootstrap and get samples
        self.samples = self._get_samples(self._nboot)

        # Get parameters for z0 and alpha
        self.z0, self.a = self._get_bca_params(self.estimate, 
                                self.jacknife, self.samples)

    def __str__(self):
        str = ' ** Bootstrap statistic **\n'
        str += '   nb variables = %d\n'%self.nvar
        str += '   nb bootstrap samples = %d\n'%self._nboot
        str += '   estimate  = %s\n'%self.estimate

        return str

    def _get_samples(self, nboot):
        ''' Get the bootstrap samples '''
        samples = np.empty((self.nvar, nboot), float)
        sh = self._data.shape
        for i in range(nboot):
            kk = np.random.choice(range(self.ndatapts), self.ndatapts)
            if len(sh)==1:
                datab = self._data[kk]
            else:
                datab = self._data[kk, :]

            samples[:,i] = self._statistic(datab)

        return samples

    def get_ci_percent(self, endpoint=0.975):
        ''' Returns bootstrap percentile confidence interval 
            assuming independence between variables
        '''
        output = np.empty((self.nvar, 3), float)
        output[:,0] = self.estimate
        for i in range(self.nvar):
            w = self.samples[i,:]
            output[i,1] = np.percentile(w, (1-endpoint)*100)
            output[i,2] = np.percentile(w, endpoint*100)

        output = pd.DataFrame(output)
        output.columns = ['estimate', 
                'confint_%3.3d'%((1-endpoint)*1000),
                'confint_%3.3d'%(endpoint*1000)]

        return output

    def _get_jacknife(self):
        ''' Get the jacknife samples '''
        jack = np.empty((self.nvar, self.ndatapts), float)
        sh = self._data.shape
        for i in range(self.ndatapts):
            kk = range(self.ndatapts)
            kk.pop(i)
            if len(sh)==1:
                dataj = self._data[kk]
            else:
                dataj = self._data[kk, :]

            jack[:,i] = self._statistic(dataj)

        return jack

    def _get_bca_params(self, estimate, jacknife, samples):
        ''' Get the BC alpha parameters  
            According to Efron and Tibshirani (1994, Chapman and Hall)
            page 186, Eq 14.14 and Eq. 14.15

        '''
        nv = self.nvar
        npt = self.ndatapts

        # Acceleration
        mj = np.mean(jacknife, axis=1).reshape((nv, 1))
        mj2 = np.sum((jacknife-np.dot(mj, np.ones((1, npt))))**2, 
                        axis=1)
        mj3 = np.sum((jacknife-np.dot(mj, np.ones((1, npt))))**3, 
                        axis=1)
        a = (-mj3/(mj2)**1.5/6).reshape((nv,))
        
        # Bias correction 
        z0 = np.empty((nv,), float)
        for i in range(nv):
            pp = np.sum(samples[i,:]<estimate[i])
            z0[i] = norm.ppf((0. + pp)/self._nboot)

        return z0, a

    def get_ci_bca(self, endpoint=0.975):
        ''' Returns bootstrap percentile confidence interval 
            assuming independence between variables
        '''
        # BC alpha parameters
        a = self.a
        z0 = self.z0

        # Compute confidence intervals
        output = np.empty((self.nvar, 3), float)
        output[:,0] = self.estimate
        for i in range(self.nvar):
            # lower bound
            u = z0[i]+norm.ppf(1-endpoint)
            v = 1-a[i]*u
            endpoint1 = norm.cdf(z0[i]+u/v)

            # upper bound
            u = z0[i]+norm.ppf(endpoint)
            v = 1-a[i]*u
            endpoint2 = norm.cdf(z0[i]+u/v)
 
            w = self.samples[i,:]
            output[i,1] = np.percentile(w, endpoint1*100)
            output[i,2] = np.percentile(w, endpoint2*100)

        output = pd.DataFrame(output)
        output.columns = ['estimate', 
                'confint_%3.3d'%((1-endpoint)*1000),
                'confint_%3.3d'%(endpoint*1000)]

        return output


