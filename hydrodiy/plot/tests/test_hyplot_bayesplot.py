import os, math

import unittest
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvt

from hydrodiy.plot import mcmcplot, putils

np.random.seed(0)

class MCMCPlotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MCMCPlotTestCase (hyplot)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        self.nchains = 5
        self.nparams = 10
        self.nsamples = 5000
        self.samples = np.random.uniform(0, 1,\
                        (self.nchains, self.nparams, self.nsamples))


        def logpost(params):
            ''' Mvt logpost '''
            mu = params[:self.nparams]
            loglike = mvt.logpdf(self.samples, mean=mu, cov=cov)


    def test_slice2d(self):
        ''' Plot log post slice 2d '''

        fig, ax = putils.get_fig_axs()
        mcmcplot.slice2d(


if __name__ == "__main__":
    unittest.main()
