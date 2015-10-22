import re
from math import sqrt
import numpy as np
import pandas as pd
import c_hystat

def percentiles(x, perc=np.linspace(0, 100, 5), prob_cst=0.3):
    ''' Compute percentiles

    Parameters
    -----------
    x : numpy.ndarray
        Input variable
    perc : list
        Percentiles to be computed
    prob_cst : float
        Probability constant to be used for empirical frequency 
        computation

    Returns
    -----------
    pp : pandas.Series
        Percentile values

    Example
    -----------
    >>> import numpy as np
    >>> from hystat import sutils
    >>> nval = 1000
    >>> x = np.random.normal(size=nval)
    >>> pp = sutils.percentiles(x)
    
    '''

    perc = np.atleast_1d(np.array(perc))

    try:
        xx = x.astype(float)
        idx = ~np.isnan(xx)
        xs = np.sort(xx[idx]).astype(np.float)
        ff = empfreq(len(xx[idx]), prob_cst)
        qq = np.interp((perc+0.)/100, ff, xs, 
                left=np.nan, right=np.nan)
        
    except TypeError:
        qq = np.nan * np.zeros(len(perc))

    idx = ['P%3.3d.%1d'%(int(p), int(10*(p-int(p)))) for p in perc]

    pp = pd.Series(qq, index=idx)

    return pp



def empfreq(nval, prob_cst=0.3):
    ''' Compute empirical frequencies for sample of size nval 
    
        :param int nval: Sample size
        :param float prob_cst: Constant used to compute frequency
    '''
    delta = 1./(nval+1-2*prob_cst)
    return (np.arange(1, nval+1)-prob_cst)*delta


def acf(data, lag=range(1,6), 
        filter=None, 
        min_val=-np.inf, 
        same_summary_stats=True):
    ''' 
        Compute lagged correlation with missing data 

        :param numpy.array data: data used to compute acf
        :param list lag: lag values
        :param numpy.array filter: boolean vector to keep only certain values in the data vector 
            CAUTION values are NOT removed from the lagged vector
            this parameter is not intended to flag missing data
        :param float min_val: Minimum value under which data is considered missing
        :param bool same_summary_stats: Use same summary stats (mean/std) for data and lagged data. True is the value used in R. p
            This can make a difference for short records
            CAUTION same_summary_stats is set to False if filter is not None
            (the acf becomes a cross correlation)
    '''
    assert np.prod(data.shape)==len(data)
    data = np.array(data).reshape((len(data)))

    out = pd.DataFrame({'acf':[0.]*len(lag),
                            'covar':np.nan,
                            'mean':np.nan, 'mean_lag':np.nan,  
                            'std':np.nan, 'std_lag':np.nan, 'nval':0}, 
                            index=lag)

    # R stat limits
    # clim0 <- if (with.ci) qnorm((1 + ci)/2)/sqrt(x$n.used) else c(0, 0)
    #clim0 * sqrt(cumsum(c(1, 2*x$acf[-1, i, j]^2))) else clim0
    
    # Summary stats for the data set
    idx0 = ((data>=min_val) & (~np.isnan(data))).flatten()
    if not filter is None:
        assert len(filter)==len(data)
        same_summary_stats = False
        idx0 = idx0 & filter
    mean = np.mean(data[idx0])
    std = sqrt(np.mean((data[idx0]-mean)**2))
    nval = np.sum(idx0)

    # loop through lags
    for l in lag:
        # Lagged data
        data_lag = np.roll(data, l)
        data_lag[:l] = np.nan

        # Filter data
        idx = idx0 & ((data_lag>=min_val) & (~np.isnan(data_lag))).flatten()
        nval_lag = np.sum(idx)

        # Summary stats
        mean1 = np.mean(data[idx])
        mean2 = np.mean(data_lag[idx])
        std1 = sqrt(np.mean((data[idx]-mean1)**2))
        std2 = sqrt(np.mean((data_lag[idx]-mean2)**2))
        if same_summary_stats: 
            std1 = std
            std2 = std
            mean1 = mean
            mean2 = mean
            nval_lag = nval

        out.loc[l, 'nval'] = nval_lag
        out.loc[l, 'mean'] = mean1
        out.loc[l, 'mean_lag'] = mean2
        out.loc[l, 'std'] = std1
        out.loc[l, 'std_lag'] = std2

        out.loc[l, 'covar'] = np.sum((data[idx]-mean1) * (data_lag[idx]-mean2))
        out.loc[l, 'covar'] /= nval_lag
        out.loc[l, 'acf'] = out['covar'][l]/std1
        out.loc[l, 'acf'] /= std2

    return out

def ar1innov(params, innov):
    ''' Compute AR1 time series from innovation 

    Parameters
    -----------
    params : list
        Parameter vector with
        params[0] ar1 coefficient
        params[1] Initial value of the innovation
    innov : numpy.ndarray
        Innovation time series 

    Returns
    -----------
    data : numpy.ndarray
        Time series of innovations

    Example
    -----------
    >>> import numpy as np
    >>> from hystat import sutils
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov([0.95, 0.], innov1)
    >>> innov2 = sutils.ar1inverse([0.95, 0.], data)
    >>> np.allclose(innov1, innov2)
    True
    
    '''

    innov = innov.reshape((np.prod(innov.shape),))

    data = np.zeros(innov.shape[0], float)

    p = np.array(params).reshape((len(params),))

    ierr = c_hystat.ar1innov(p, innov, data)

    if ierr!=0:
        raise ValueError('ar1innov returns %d'%ierr)

    return data


def ar1inverse(params, data):
    ''' Compute innovations from an AR1 time series 

    Parameters
    -----------
    params : list
        Parameter vector with
        params[0] ar1 coefficient
        params[1] Initial value of the innovation
    data : numpy.ndarray
        Time series of ar1 data

    Returns
    -----------
    innov : numpy.ndarray
        Time series of innovations

    Example
    -----------
    >>> import numpy as np
    >>> from hystat import sutils
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov([0.95, 0.], innov1)
    >>> innov2 = sutils.ar1inverse([0.95, 0.], data)
    >>> np.allclose(innov1, innov2)
    True

    ''' 

    innov = np.zeros(data.shape[0], float)

    p = np.array(params).reshape((len(params),))

    ierr = c_hystat.ar1inverse(p, data, innov)

    if ierr!=0:
        raise ValueError('c_hystat.ar1inverse returns %d'%ierr)

    return innov


def pit(obs, forc):
    ''' 
        Compute PIT for ensemble forecasts

        :param numpy.array obs Observed data
        :param numpy.array forc Forecast data
    '''

    nval = len(obs)
    nens = forc.shape[1]

    if forc.shape[0] != nval:
        raise ValueError('forc.shape[1](%d) != len(obs)(%d)' % (forc.shape[0], nval))

    pit = np.ones_like(obs) * np.nan
    ff = empfreq(forc.shape[1])

    for i in range(nval):

        if pd.notnull(obs[i]):

            # Add small perturbation to avoid ties
            f = np.sort(forc[i, :])

            d = np.abs(np.diff(f))
            idx = d>0
            eps = 1e-20
            if np.sum(idx)>0:
                eps = np.min(d[d>0]) * 1e-10

            e = np.random.uniform(0, eps, nens)
            f = f + e

            # Compute PIT
            pit[i] = np.interp(obs[i], f, ff, left=0, right=1)
        

    return pit


def lhs(nparams, nsample, pmin, pmax, seed=0):
    ''' Latin hypercube sampling

    Parameters
    -----------
    nparams : int
        Number of parameters
    nsample : int
        Number of sample to draw
    pmin : list
        Lower bounds of parameters
    pmax : list
        Upper bounds of parameters
    seed : int
        Random seed

    Returns
    -----------
    samples : numpy.ndarray
        Parameter samples (nsamples x nparams)

    Example
    -----------
    >>> import numpy as np
    >>> from hystat import sutils
    >>> nparams = 5; nsamples = 3
    >>> sutils.lhs(nparams, nsamples)
    
    '''

    if not isinstance(pmin, list):
        pmin = [pmin] * nparams

    if not isinstance(pmax, list):
        pmax = [pmax] * nparams

    samples = np.zeros((nsample, nparams))

    for i in range(nparams):
        du = float(pmax[i]-pmin[i])/nsample
        u = np.linspace(pmin[i]+du/2, pmax[i]-du/2, nsample)

        kk = np.random.permutation(nsample)
        s = u[kk] + np.random.uniform(-du/2, du/2, size=nsample)

        samples[:, i] = s

    return samples

def schaakeshuffle(obs, forc_input, eps=1e-30, copy=True):
    ''' Apply the Schaake shuffle technique to ensemble forecasts
    The function does not returns a value.
    It only reshuffles the forc array.

    Parameters
    -----------
    obs : numpy.ndarray
        Observation data with
        N rows representing N concomittant occurence of data
        P columns representing P variables
    forc : numpy.ndarray
        Ensemble forecast with
        M rows representing M ensemble members
        P columns representing P variables
    eps : float
        Small quantity added to observed data in case where N < M
    copy: bool
        Creates a copy of the forecast input
        If set to false, then the forc_input array gets reshuffled in place

    Returns
    -----------
    forc : numpy.ndarray
        Reshuffled forecasts. Array NxP

    Example
    -----------
    >>> import numpy as np
    >>> import schaakeshuffle
    >>> # Correlated observations
    >>> nobs = 30
    >>> mu = [10, 30]
    >>> sig = [[25, 45], [45, 100]]
    >>> obs = np.random.multivariate_normal(mu, sig, nobs)
    >>> # Uncorrelated forecasts
    >>> nens = 1000
    >>> sig = [[25, 0], [0, 100]]
    >>> forc = np.random.multivariate_normal(mu, sig, nens)
    >>> schaakeshuffle.schaakeshuffle(obs, forc)

    '''

    if copy:
        forc = forc_input.copy()
    else:
        forc = forc_input

    if len(obs.shape) != 2:
        raise ValueError('if len(obs.shape)({0}) != 2'.format(len(obs.shape)))

    if len(forc.shape) != 2:
        raise ValueError('if len(forc.shape)({0}) != 2'.format(len(forc.shape)))

    if obs.shape[1] != forc.shape[1]:
        raise ValueError('obs.shape[1]({0}) != forc.shape[1]({1})'.format(
            obs.shape[1], forc.shape[1]))

    if obs.shape[0] > forc.shape[0]:
        raise ValueError('obs.shape[0]({0}) > forc.shape[0]({1}'.format(
            obs.shape[0], forc.shape[0]))

    # Get dimensions
    nvar = obs.shape[1]
    nobs = obs.shape[0]
    nens = forc.shape[0]

    # Resample obs if nobs < nens
    obs2 = obs

    if nens > nobs:
        obs2 = np.zeros_like(forc)

        for i in range(nens):
            k = (i * nobs)/nens
            obs2[i,:] = obs[k,:]

        obs2 += np.random.uniform(0, eps, size=(nens,1))

    # Shuffle ensembles
    for j in range(nvar):
        k = np.argsort(obs2[:, j])
        forc[k, j] = np.sort(forc[:, j])


    return forc


