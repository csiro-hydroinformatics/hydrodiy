import os
import unittest
import numpy as np
from math import log, exp, sqrt

from scipy.linalg import cholesky as chol
import pandas as pd
from numpy.testing import assert_array_almost_equal as almost_equal

import matplotlib.pyplot as plt

from hystat import mcmc

class McmcTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> McmcTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

        # Log posterior of normal distribution
        # for the unknown mean and known variance
        # with uniformative (i.e. constant) prior
        def logPost(pars, data):
            n = data.shape[0]
            lp = -pars[1]*(n+1)
            lp += -np.sum((data-pars[0])**2)/exp(pars[1])**2/2
            return lp
        
        # Sample normal population
        mu = 5
        sigma = 4
        nval = 50
        self.normal_data = {'mu':mu, 'sigma':sigma, 
                'data':np.random.normal(loc=mu, 
                        scale=sigma, size=nval),
                'logPost':logPost}
        
    def test_mvt(self):
        
        np.random.seed(333)

        # define distribution
        mu = np.array([1., 2.])
        cov = np.array([[10., 2.], [2., 3.]])
        Lcov = chol(cov, lower=True)
        
        # Sample population
        nsamp = int(1e5) 
        sample = np.empty((nsamp, 2), float)
        for i in range(nsamp):
            sample[i,:] = mcmc.mvt(mu, Lcov)

        # Compare mean and covariance
        mu2 = np.mean(sample, axis=0)
        cond = almost_equal(mu, mu2, 2)
        self.assertTrue(cond is None)

        cov2 = np.cov(sample.T)
        cond = almost_equal(cov, cov2, 1)
        self.assertTrue(cond is None)

    def test_metro_runner(self):
        ''' Not a formal test. Just check mcmc_runner works '''
        np.random.seed(333)
        data = self.normal_data

        # starting points and proposal distribu
        prop_means = np.array([data['mu']+1.0, 
                        log(data['sigma'])+0.2])
        prop_cov = np.array([[5., 1.], [1., 5.]])
       
        # Run MCMC
        fargs = (data['data'], )
        sampler = mcmc.metro_runner(data['logPost'],
                prop_means, prop_cov, 
                ndisplay=0, 
                nchains=5,
                nsample=10, 
                fargs=fargs)
        
        dtype = mcmc.dtype_chain(2)
        sample = np.empty(0, dtype=dtype)
        for smp in sampler:
            sample = np.hstack([sample, smp])
        
    def test_metro_mcmc(self):
        ''' Test of estimation procedure for normal data '''
        np.random.seed(333)
        data = self.normal_data

        # proposal distribution
        nv = len(data['data'])
        mu_p = np.abs(data['mu'])/sqrt(nv)
        lsig_p = (1.+np.abs(log(data['sigma'])))/sqrt(nv)
        prop_cov = np.array([[mu_p, 0.0], [0.0, lsig_p]])
       
        # Run MCMC
        start = np.array([0. , 0.])
        logPost = data['logPost']
        fargs = (data['data'], )

        sample, prop_mean, prop_cov = mcmc.metro_mcmc(logPost, 
                start, ndisplay=0, 
                nchains=5, 
                nsample=1000,
                fargs=fargs,
                par_names_short=['mu', 'logsigma'])

        fsample = '%s/mcmc_sample.csv'%self.FOUT 
        mcmc.write_chain_data(sample, fsample, 
                'Inference of mean and variance from normal data')

    def test_plot(self):
        ''' Test of plotting function '''
        sample = mcmc.read_chain_data('%s/mcmc_sample.csv.gz'%self.FOUT)

        data = self.normal_data

        # configure plot
        plt.close()
        fig = plt.figure(figsize=(16,10))
        fig.subplots_adjust(wspace=0.4, hspace=0.3)

        # Generate MCMC plots
        axs, gs = mcmc.plot_chains(sample, fig, nval_trace=100)
        
        # Add more data
        ax = axs['trace_param_mu']
        ti = '%s (mu = %0.2f)'%(ax.get_title(), data['mu'])
        ax.set_title(ti)
        xlim = ax.get_xlim()
        ax.plot(xlim, [data['mu']]*2, 'k-', lw=3)
        ax.plot(xlim, [np.mean(data['data'])]*2, 'k--', lw=3)

        ax = axs['trace_param_logsigma']
        ti = '%s (log(sigma) = %0.2f)'%(ax.get_title(), 
                                        log(data['sigma']))
        ax.set_title(ti)
        xlim = ax.get_xlim()
        ax.plot(xlim, [log(data['sigma'])]*2, 'k-', lw=3)
        ax.plot(xlim, [sqrt(np.var(data['data']))]*2, 'k--', lw=3)

        ax = fig.add_subplot(gs[0,3])
        nb = int(len(data['data'])/5)
        ax.hist(data['data'], bins=nb, color='blue', alpha=0.8)
        ax.set_title('Original normal data / nval=%d'%len(data['data']))
        ylim = ax.get_ylim()
        ax.plot([data['mu']]*2, ylim, 'k-', lw=3)
        ax.plot([np.mean(data['data'])]*2, ylim, 'k--', lw=3)

        #plt.show()
        #import pdb; pdb.set_trace()
        plt.savefig('%s/mcmc_plot.png'%self.FOUT)

if __name__ == "__main__":
    unittest.main()
