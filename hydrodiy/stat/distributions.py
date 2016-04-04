import math
import numpy as np

from  scipy.misc.doccer import inherit_docstring_from
from scipy.stats import rv_continuous, norm

from scipy.optimize import fmin_powell as fmin


class LogNormShiftedCensored(rv_continuous):
    ''' Log-Normal distribution with left censoring and shifting '''

    def __init__(self, censor=0.):

        if not np.isfinite(censor):
            raise ValueError('Censor ({0}) should be finite'.format(censor))

        rv_continuous.__init__(self,
            name='lognormscensored',
            longname='Left censored log-normal distribution with shifting parameter',
            shapes='mu,sig,shift',
            a=censor)


    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.isfinite(np.asarray(arg))))
        return cond


    def _pdf(self, x, mu, sig, shift):
        censor = self.a
        if shift <= -censor:
            raise ValueError('shift({0}) <= -censor ({1})'.format(shift,
            -censor))

        pp = norm.pdf((np.log(x+shift)-mu)/sig)/(x+shift)
        pp[np.isclose(x, censor)] = np.inf
        return pp


    def _cdf(self, x, mu, sig, shift):
        censor = self.a
        if shift <= -censor:
            raise ValueError('shift({0}) <= -censor ({1})'.format(shift,
            -censor))

        cp = norm.cdf((np.log(x+shift)-mu)/sig)
        return cp


    def _ppf(self, q, mu, sig, shift):
        censor = self.a
        if shift <= -censor:
            raise ValueError('shift({0}) <= -censor ({1})'.format(shift,
            -censor))

        P0 = norm.cdf((np.log(shift)-mu)/sig)
        qq = q*(1-P0)+P0
        x = np.exp(sig*norm.ppf(qq)+mu)-shift
        return x


    def _fitstart(self, data):
        data = np.asarray(data)
        lm = np.log(data[data>0.])
        mu = np.nanmean(lm)
        sig = math.sqrt(np.mean((lm-mu)**2))
        return mu, sig, np.nan


    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwargs):
        data = np.asarray(data)

        # Get censor
        censor = self.a

        # Determines frequency of values below censor
        nval = len(data)
        idx0 = ~(data>censor)
        idx1 = ~idx0
        n0 = np.sum(idx0)

        # Log posterior
        def loglikelihood(params):
            # get parameters
            mu = params[0]
            logsig = params[1]
            sig = math.exp(params[1])

            shift = -censor + math.exp(params[2])
            lx = np.log(shift+data[idx1])
            err = (lx-mu)/sig
            ll = (nval-n0)*logsig + np.sum(lx+err*err/2)

            P0 = norm.cdf((math.log(shift+censor)-mu)/sig)
            if P0 > 0:
                ll -= n0*math.log(P0)

            if np.isnan(ll):
                raise ValueError('Nan value returned by likelihood')

            return ll

        # Start parameters
        mu, sig, _ = self._fitstart(data)
        params0 = [mu, math.log(sig), math.log(1e-30+censor)]

        # Optimization
        params, fopt, direc, niter, feval, _ = fmin(loglikelihood,
            params0, maxfun=1000, xtol=1e-5, ftol=1e-5,
            full_output=True, disp=False)

        mu = params[0]
        sig = math.exp(params[1])
        shift = -censor + math.exp(params[2])

        return (mu, sig, shift, 0., 1.)


lognormscensored0 = LogNormShiftedCensored(censor=0.)

