import re
from math import sqrt
import numpy as np
import pandas as pd
import _sutils

def percentiles(x, perc=np.linspace(0, 100, 5), prob_cst=0.3):
    ''' Returns percentiles of the input variable as a Pandas Series.
        Returns a series filled up with nans if the input type is wrong

        :param np.array x: Sample
        :param list perc: Percentile values
        :param float prob_cst: Constant used to compute frequency
    '''
    perc = np.atleast_1d(np.array(perc))
    try:
        xs = np.sort(x)
        ff = empfreq(len(x), prob_cst)
        qq = np.interp((perc+0.)/100, ff, xs, left=np.nan, right=np.nan)
        
    except TypeError:
        qq = np.nan * np.zeros(len(perc))

    idx = [re.sub(' ', '_', 'P%5.1f'%p) for p in perc]

    return pd.Series(qq, index=idx)

def categories(x, 
        bounds=np.linspace(0, 100, 5), 
        is_percentile=True, 
        has_ties=True,
        format_cat='[%4.1f,\n%4.1f['):
    '''
        Compute categorical data from continuous data

        :param np.array x: data from continuous variable  
        :param bounds x: boundaries of categories
        :param bool is_percentile: consider bounds as percentage 
                    used to compute percentiles
        :param bool has_ties: Are there ties in x value (e.g. 0.0) ?
        :param string format_cat: format to use for printing categories

        :return 
    '''

    # remove ties in x variable
    y = x 
    if has_ties:
        dy = np.max(np.diff(np.sort(y),1))
        y += np.random.uniform(size=len(y))*dy*1e-6

    # Compute boundaries 
    yq = bounds
    if is_percentile:
        yq = percentiles(y, bounds, 1.)
        
        # add a small number to include max
        yq[len(bounds)-1] += 1e-10 

    yq = np.unique(np.sort(yq))

    # default catval is np.nan
    catval = np.array([-1]*len(y), dtype='int')
    catnames = np.array(['NA']*(len(yq)-1), dtype='S50')

    # compute categories
    for i in range(len(yq)-1):
        idx = (y>=yq[i]) & (y<yq[i+1])
        catval[idx] = i
        catnames[i] = format_cat%(yq[i], yq[i+1])

    return catval, catnames

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

        :param np.array data: data used to compute acf
        :param list lag: lag values
        :param np.array filter: boolean vector to keep only certain values in the data vector 
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

        out['nval'][l] = nval_lag
        out['mean'][l] = mean1
        out['mean_lag'][l] = mean2
        out['std'][l] = std1
        out['std_lag'][l] = std2

        out['covar'][l] = np.sum((data[idx]-mean1) * (data_lag[idx]-mean2))
        out['covar'][l] /= nval_lag
        out['acf'][l] = out['covar'][l]/std1
        out['acf'][l] /= std2

    return out

def ar1random(params, nval, seed=0):
    ''' 
        Run ar1 model with normal innovation

        :param np.array params: parameter vector with
            params[0] = ar1 parameter
            params[1] = sigma of innovation (used if innov is None)
            params[2] = output value at t=0
        :param int nval: Number of values
        :param int seed: Random generator seed number
    '''

    output = np.zeros(nval, float)
    ierr = _sutils.ar1random(np.array(params), seed, output)

    if ierr!=0:
        raise ValueError('ar1random returns %d'%ierr)

    return output

def ar1innov(params, innov):
    ''' 
        Run ar1 model with normal innovation

        :param np.array params: parameter vector with
            params[0] = ar1 parameter
            params[1] = output value at t=0
    '''

    output = np.zeros(innov.shape[0], float)
    ierr = _sutils.ar1innov(params, innov, output)

    if ierr!=0:
        raise ValueError('ar1innov returns %d'%ierr)

    return output

def ar1inverse(params, input):
    ''' 
        Run ar1 model with normal innovation

        :param np.array params: parameter vector with
            params[0] = ar1 parameter
    '''

    innov = np.zeros(input.shape[0], float)
    ierr = _sutils.ar1inverse(params, input, innov)

    if ierr!=0:
        raise ValueError('ar1inverse returns %d'%ierr)

    return innov


