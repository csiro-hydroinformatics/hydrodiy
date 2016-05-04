import math
import numpy as np

from  scipy.misc.doccer import inherit_docstring_from
from scipy.stats import rv_continuous, norm
from scipy.stats import mannwhitneyu, ks_2samp

from scipy.optimize import fmin_powell as fmin

from hydrodiy.stat import sutils


class LogNormShiftedCensored(rv_continuous):
    ''' Log-Normal distribution with left censoring and shifting '''

    def __init__(self, censor=0., maxexponential=2e1):

        if not np.isfinite(censor):
            raise ValueError('Censor ({0}) should be finite'.format(censor))

        rv_continuous.__init__(self,
            name='lognormscensored',
            longname='Left censored log-normal distribution with shifting parameter',
            shapes='mu,sig,shift',
            a=censor)

        self.fit_diagnostic = {}
        self.maxexponential = maxexponential


    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.isfinite(np.asarray(arg))))
        return cond


    def _pdf(self, x, mu, sig, shift):
        censor = self.a
        if np.any(shift <= -censor):
            raise ValueError('shift <= -censor ({1})'.format(-censor))

        pp = norm.pdf((np.log(x+shift)-mu)/sig)/(x+shift)
        pp[x<censor] = 0.
        pp[np.isclose(x, censor)] = np.inf
        return pp


    def _cdf(self, x, mu, sig, shift):
        censor = self.a
        if np.any(shift <= -censor):
            raise ValueError('shift <= -censor ({1})'.format(-censor))

        cp = norm.cdf((np.log(x+shift)-mu)/sig)
        cp[x<censor] = 0.
        return cp


    def _ppf(self, q, mu, sig, shift):
        censor = self.a
        if np.any(shift <= -censor):
            raise ValueError('shift <= -censor ({0})'.format(-censor))

        x = np.exp(sig*norm.ppf(q)+mu)-shift
        x[x<censor] = censor
        return x


    def _fitstart(self, data):
        data = np.asarray(data)
        lm = np.log(data[data>0.])
        mu = np.nanmean(lm)
        sig = math.sqrt(np.mean((lm-mu)**2))
        return mu, sig, np.nan


    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwargs):
        data = np.asarray(data, dtype=float)
        data = np.sort(data)

        # Get censor
        censor = self.a

        # Determines frequency of values below censor
        nval = len(data)
        n0 = np.min(np.where(data > censor)[0])

        # Log posterior
        mexp = self.maxexponential

        def loglikelihood(params):

            if not (abs(params[1])<mexp and abs(params[2])<mexp):
                return np.inf

            # get parameters
            mu = params[0]
            logsig = params[1]
            sig = math.exp(logsig)

            shift = -censor + math.exp(params[2])
            lx = np.log(shift+data[n0:])
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

        # multi start optimization
        fopt = np.inf
        nstart = 5

        for i, a0 in enumerate(np.linspace(-mexp, math.log(abs(mu)), nstart)):
            params0 = [mu, math.log(sig), a0]

            tmp_params, tmp_fopt, direc, niter, feval, _ = fmin(loglikelihood,
                params0, maxfun=1000, xtol=1e-5, ftol=1e-5,
                full_output=True, disp=False)

            #tmptxt = ' '.join(['{0:3.3e}'.format(pp) for pp in params0])
            #print('Start {0}: ll={1:3.3e} p=[{2}] n={3} nf={4}'.format(i+1,
            #    tmp_fopt, tmptxt, niter, feval))

            if tmp_fopt < fopt:
                params = tmp_params.copy()
                fopt = tmp_fopt

        mu = params[0]
        sig = math.exp(params[1])
        shift = -censor + math.exp(params[2])

        # Diagnostic
        ff = sutils.empfreq(nval)
        sim = self.ppf(ff, mu, sig, shift)
        ks, kspv = ks_2samp(data, sim)
        mw, mwpv = mannwhitneyu(data, sim)

        self.fit_diagnostic = {'loglikelihood':fopt,
            'params':params,
            'ks_stat': ks,
            'ks_pvalue': kspv,
            'mw_stat': mw,
            'mw_pvalue': mwpv,
        }

        return (mu, sig, shift, 0., 1.)


lognormscensored0 = LogNormShiftedCensored(censor=0.)


def power(x, lam, cst, tol=1e-10):
    ''' Power function |x|/x * (|x|^lam-cst^lam)/lam'''
    xa = cst+np.abs(x)
    xs = np.sign(x)

    if lam<0:
        raise ValueError('lam<0')

    if cst<=tol:
        raise ValueError('cst<=0')

    if abs(lam)>tol:
        y = xs * (xa**lam-cst**lam)/lam
    else:
        y = xs * np.log(xa/cst)

    return y

def powerdiff(x, lam, cst, tol=1e-10):
    ''' Derivative of power function '''
    xa = cst+np.abs(x)
    xs = np.sign(x)

    if lam<0:
        raise ValueError('lam<0')

    if cst<=tol:
        raise ValueError('cst<=0')

    if abs(lam)>tol:
        dy = xa**(lam-1)
    else:
        dy = cst/xa

    return dy

def powerinv(y, lam, cst, tol=1e-10):
    ''' Inverse power function '''
    ya = np.abs(y)
    ys = np.sign(y)

    if lam<0:
        raise ValueError('lam<0')

    if cst<=tol:
        raise ValueError('cst<=0')

    if abs(lam)>tol:
        x = ys*((lam*ya+cst**lam)**(1./lam)-cst)
    else:
        x = ys*(np.exp(ya)*cst-cst)

    return x

class PowerNormCensored(rv_continuous):
    ''' Power transform Normal distribution with left censoring '''

    def __init__(self, censor=0., cst=1e-3, maxexponential=2e1):

        if not np.isfinite(censor):
            raise ValueError('Censor ({0}) should be finite'.format(censor))

        rv_continuous.__init__(self,
            name='powernormcensored',
            longname='Left censored power transform normal distribution',
            shapes='mu,sig,lam',
            a=censor)


        if cst <= 0:
            raise ValueError('cst({0}) not >0'.format(cst))
        self.cst = cst

        self.fit_diagnostic = {}
        self.maxexponential = maxexponential


    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.isfinite(np.asarray(arg))))
        return cond


    def _pdf(self, x, mu, sig, lam):
        censor = self.a
        cst = self.cst
        if np.any(lam < 0 or lam >=3):
            raise ValueError('lam({0}) not in [0, 3['.format(lam))


        y = power(x, lam, cst)
        dy = powerdiff(x, lam, cst)
        pp = norm.pdf((y-mu)/sig)*dy
        pp[x<censor] = 0.
        pp[np.isclose(x, censor)] = np.inf
        return pp


    def _cdf(self, x, mu, sig, lam, cst):
        censor = self.a
        cst = self.cst
        if np.any(lam < 0 or lam >=3):
            raise ValueError('lam({0}) not in [0, 3['.format(lam))

        y = power(x, lam, cst)
        cp = norm.cdf((y-mu)/sig)
        cp[x<censor] = 0.
        return cp


    def _ppf(self, q, mu, sig, lam, cst):
        censor = self.a
        cst = self.cst
        if np.any(lam < 0 or lam >=3):
            raise ValueError('lam({0}) not in [0, 3['.format(lam))

        x = powerinv(sig*norm.ppf(q)+mu, lam, cst)
        x[x<censor] = censor
        return x


    def _fitstart(self, data):
        pass
        #data = np.asarray(data)
        #lm = np.log(data[data>0.])
        #mu = np.nanmean(lm)
        #sig = math.sqrt(np.mean((lm-mu)**2))
        #return mu, sig, np.nan


    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwargs):
        data = np.asarray(data, dtype=float)
        data = np.sort(data)

        # Get censor and constant
        censor = self.a
        cst = self.cst

        # Determines frequency of values below censor
        nval = len(data)
        n0 = np.min(np.where(data > censor)[0])

        ## Log posterior
        mexp = self.maxexponential

        def loglikelihood(params):

            if not (abs(params[1])<mexp and abs(params[2])<mexp):
                return np.inf

            # get parameters
            mu = params[0]
            logsig = params[1]
            sig = math.exp(logsig)

            e = math.exp(params[2])
            lam = 3*e/(1+e)

            y = power(data[n0:], lam, cst)
            err = (y-mu)/sig
            # TODO !
            #ll = (nval-n0)*logsig + np.sum(lx+err*err/2)

            P0 = norm.cdf((power(censor, lam, cst)-mu)/sig)
            if P0 > 0:
                ll -= n0*math.log(P0)

            if np.isnan(ll):
                raise ValueError('Nan value returned by likelihood')

            return ll

        ## Start parameters
        #mu, sig, _ = self._fitstart(data)

        ## multi start optimization
        #fopt = np.inf
        #nstart = 5

        #for i, a0 in enumerate(np.linspace(-mexp, math.log(abs(mu)), nstart)):
        #    params0 = [mu, math.log(sig), a0]

        #    tmp_params, tmp_fopt, direc, niter, feval, _ = fmin(loglikelihood,
        #        params0, maxfun=1000, xtol=1e-5, ftol=1e-5,
        #        full_output=True, disp=False)

        #    #tmptxt = ' '.join(['{0:3.3e}'.format(pp) for pp in params0])
        #    #print('Start {0}: ll={1:3.3e} p=[{2}] n={3} nf={4}'.format(i+1,
        #    #    tmp_fopt, tmptxt, niter, feval))

        #    if tmp_fopt < fopt:
        #        params = tmp_params.copy()
        #        fopt = tmp_fopt

        #mu = params[0]
        #sig = math.exp(params[1])
        #shift = -censor + math.exp(params[2])

        ## Diagnostic
        #ff = sutils.empfreq(nval)
        #sim = self.ppf(ff, mu, sig, shift)
        #ks, kspv = ks_2samp(data, sim)
        #mw, mwpv = mannwhitneyu(data, sim)

        #self.fit_diagnostic = {'loglikelihood':fopt,
        #    'params':params,
        #    'ks_stat': ks,
        #    'ks_pvalue': kspv,
        #    'mw_stat': mw,
        #    'mw_pvalue': mwpv,
        #}

        #return (mu, sig, shift, 0., 1.)

powernormcensored0 = PowerNormCensored(censor=0.)

