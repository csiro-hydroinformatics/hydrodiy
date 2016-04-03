import math
import numpy as np

from  scipy.misc.doccer import inherit_docstring_from
from scipy.stats import rv_continuous, norm

from scipy.optimize import fmin_powell as fmin

class LogNormCensored(rv_continuous):
    ''' Log-Normal distribution with left censoring '''

    def __init__(self, censor=0.):
        rv_continuous.__init__(self,
            name='lognormcensored',
            longname='Left censored log-normal distribution',
            a=censor)

    def __checkparams(self, shift, censor):
        if shift <= -censor:
            raise ValueError('shift({0}) <= -censor ({1})'.format(shift,
            -censor))

    def _pdf(self, x, mu, sig, shift):
        censor = self.a
        self.__checkparams(shift, censor)
        pp = norm.pdf((np.log(x+shift)-mu)/sig)/(x+shift)
        pp[np.allclose(x, censor)] = np.inf
        pp[x<censor] = 0.
        return y

    def _cdf(self, x, mu, sig, shift):
        censor = self.a
        self.__checkparams(shift, censor)
        cp = norm.cdf((np.log(x+shift)-mu)/sig)
        cp[x<censor] = 0.
        return y

    def _ppf(self, q, mu, sig, shift):
        censor = self.a
        self.__checkparams(shift, censor)
        x = np.exp(sig*norm.ppf(q)+mu)-a
        x[x<censor] = censor
        return x

    def _stat(self, mu, sig, shift):

        # TODO = Fix this
        mean = 0.
        stdev = 0.
        skew = 0.
        kurt = 0.

        return mean, stdev, skew, kurt

    def _fitstart(self, data):
        data = np.asarray(data)
        lm = np.log(data)
        mu = np.nanmean(lm)
        sig = math.sqrt(np.nanmean((lm-mu)**2))
        return mu, sig, 1e-10

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

            if not np.isinf(censor):
                shift = -censor + math.exp(params[2])
            else:
                shift = math.sinh(params[2])

            # compute likelihood
            xs = shift+data[idx1]
            lxs = np.log(xs)
            err = (lxs-mu)/sig
            ll = (nval-n0)*logsig + np.sum(lxs+err*err/2)

            P0 = norm.cdf((math.log(shift+censor)-mu)/sig)
            if P0 > 0:
                ll -= n0*math.log(P0)

            return ll

        # Start parameters
        mu, sig, shift = self._fitstart(data)
        if not np.isinf(censor):
            params0 = [mu, math.log(sig), math.log(shift+censor)]
        else:
            params0 = [mu, math.log(sig), math.asinh(shift+censor)]

        # Optimization
        params, fopt, direc, niter, feval, _ = fmin(loglikelihood,
            params0, maxfun=1000, full_output=True, disp=False)

        mu = params[0]
        sig = math.exp(params[1])
        shift = -censor + math.exp(params[2])

        return (mu, sig, shift, 0., 1.)


