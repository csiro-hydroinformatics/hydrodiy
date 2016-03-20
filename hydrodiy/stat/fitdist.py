import math, itertools

import numpy as np
import pandas as pd

from scipy.stats import norm

from hydrodiy.stat import transform


def normalloglikehood(trans, xx, censor, P0, params, mu, sigma):

    trans.params = params

    # Normal likelihood
    ll = -np.sum((trans.forward(xx)-mu)**2/sigma**2)
    ll -= len(xx) * math.log(sigma)

    # Transformation jacobian
    ll += np.sum(np.log(trans.jac(xx)))

    # Normalisation constant for truncated dist
    if not np.isinf(censor):
        F0 = norm.cdf(trans.forward(censor), mu, sigma)
        ll -= np.sum([np.log(1-F0)] * len(xx))

    return -ll


class FitDist:
    ''' Fitting censored normal distribution on transformed data
        with maximum likelihood approach '''

    def __init__(self, transformation, censor=0.):
        ''' Normalise station id by removing trailing letters,
        underscores and leading zeros

        Parameters
        -----------
        transformation : str
            Transformation name from hystat.transform
        method : str
            Fitting method. 'maxlike' is maximum likelihood
            'qtm' is quantile matching
        censor : float
            censoration value
        '''
        self.trans = transform.getinstance(transformation)
        self.censor = censor

        self.mu = np.nan
        self.sigma = np.nan
        self.P0 = np.nan
        self.x_fit_notcensored = [np.nan]

        self.fitfun = normalloglikehood

    def __str__(self):
        s = ('%s\nNormal fit:\n'
                ' Censor = %0.2f\n '
                'P0 = %0.2f\n '
                'mu = %0.2f\n '
                'sigma = %0.2f') % (
                self.trans.__str__(), self.censor, self.P0,
                self.mu, self.sigma)
        return s

    def clone(self):
        f = FitDist(self.trans)
        f.trans.params = [p for p in self.trans.params]
        f.mu = self.mu
        f.sigma = self.sigma
        f.P0 = self.P0

        return f

    def fit(self, x, eps=1e-10, nexplore=100):
        ''' Perform the fitting on the data x

        Parameters
        -----------
        x : numpy.ndarray
            Observed data
        eps : float
            Threshold below which x is considered a null value

        Example
        -----------
        >>> x = np.exp(np.random.normal(loc=3, scale=2, size=100))
        >>> fd = fitdist.FitDist()
        >>> fd.fit(x)
        'A04567'
        '''

        # Remove null values
        idx = pd.notnull(x)
        x = x[idx]

        # Find values below censor
        idx = x < self.censor + eps
        self.x_fit_notcensored = x[~idx]
        self.P0 = float(np.sum(idx))/len(x)

        # Fitting function
        def errfun(pp):
            params = pp[:nparams]
            mu = pp[nparams]
            sigma = transform.bounded(pp[nparams+1], 0, 20)

            ee = self.fitfun(self.trans,
                    self.x_fit_notcensored,
                    self.censor, self.P0, params, mu, sigma)

            return ee

        # Systematic Exploration
        nparams = self.trans.nparams
        minee = np.inf
        u = np.linspace(-5, 5, nexplore)
        li = [u] * nparams

        for params in itertools.product(*li):
            self.trans.params = params
            xx = self.x_fit_notcensored
            yy = self.trans.forward(xx)

            # Compute moments
            mu = np.mean(yy)
            sigma_trans = transform.inversebounded(math.sqrt(np.var(yy)), 0, 20)

            # Build parameter vector
            p = list(params) + [mu, sigma_trans]
            ee = errfun(p)

            # Check error
            if (ee < minee) & (~np.isnan(ee)):
                minee = ee
                p0 = p

        p = p0
        ee = errfun(p)

        # Store parameters
        self.trans.params = p[:nparams]
        self.mu = p[nparams]
        self.sigma = transform.bounded(p[nparams+1], 0, 20)

        if np.isnan(self.mu):
            import pdb; pdb.set_trace()

        return {'errfun': ee,
                'params': self.trans.params,
                'mu':self.mu,
                'sigma':self.sigma}


    def sample(self, nsample):

        # censored values
        u = np.random.uniform(0, 1, size=nsample)
        sample = np.array([self.censor] * nsample)

        # non censored values
        idx = u>=self.P0
        nv = np.sum(idx)

        F0 = norm.cdf(0, self.mu, self.sigma)
        v = np.random.uniform(F0, 1, size=nv)
        y = norm.ppf(v, self.mu, self.sigma)
        x = self.trans.inverse(y) + self.censor

        sample[idx] = x

        return sample


