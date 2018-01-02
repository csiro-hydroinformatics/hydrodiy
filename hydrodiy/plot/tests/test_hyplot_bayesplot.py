import os, math

import unittest
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvt

from hydrodiy.stat import bayesutils
from hydrodiy.plot import bayesplot, putils

np.random.seed(0)

class BayesPlotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BayesPlotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        self.nchains = 3
        self.nparams = 4
        self.nsamples = 5000

        # Generate mean vector
        self.mu = np.linspace(0, 1, self.nparams)

        # Generate covariance matrix
        rnd = np.random.uniform(-1, 1, (self.nparams, self.nparams))
        rnd = rnd+rnd.T
        eig, vects = np.linalg.eig(rnd)
        self.cov = np.dot(vects, np.dot(np.diag(np.abs(eig)), vects.T))

        # Generate samples
        self.samples = np.zeros((self.nchains, self.nparams, self.nsamples))
        for chain in range(self.nchains):
            self.samples[chain, :, :] = np.random.multivariate_normal(\
                        mean=self.mu, cov=self.cov, size=self.nsamples).T

        # MVT log posterior for the first chain
        def logpost(theta):
            ''' Mvt logpost '''
            mu, cov, _, _ = bayesutils.params2mucov(theta)
            loglike = mvt.logpdf(self.samples[0, :, :].T, mean=mu, cov=cov)
            # Jeffreys' prior
            logprior = -(mu.shape[0]+1)/2*math.log(np.linalg.det(cov))
            return np.sum(loglike)+logprior

        self.logpost = logpost


    def test_slice2d(self):
        ''' Plot log post slice 2d '''

        fig, ax = putils.get_fig_axs()
        params, _, _ = bayesutils.mucov2params(self.mu, self.cov)
        bayesplot.slice2d(ax, self.logpost, params, \
                                    0, 1, 2, 2)
        fp = os.path.join(self.ftest, 'slice_2d.png')
        fig.savefig(fp)


    def test_plotchains(self):
        ''' Plot chain diagnostic '''

        fig = plt.figure()
        accept = np.ones(self.samples.shape[0])
        bayesplot.plotchains(fig, self.samples, accept)
        fp = os.path.join(self.ftest, 'plotchains.png')
        fig.set_size_inches((18, 10))
        fig.tight_layout()
        fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
