
import re, os
import pkg_resources

import numpy as np
import pandas as pd

from scipy.stats import kstest

from hydrodiy.stat import transform
from hydrodiy.stat import sutils
from hydrodiy.io import csv

import c_hydrodiy_stat

EPS = 1e-10

# Reads Cramer-Von Mises table
CVPATH = pkg_resources.resource_filename(__name__, \
                    os.path.join('data', 'cramer_von_mises_test_pvalues.zip'))
CVTABLE, _ = csv.read_csv(CVPATH, index_col=0)
CVNSAMPLE  = CVTABLE.columns.values.astype(int)
CVQQ = CVTABLE.index.values
CVTABLE = CVTABLE.values


def get_transform(name):
    ''' Get transfrom by name from transform library '''

    if name == 'Identity':
        trans = transform.Identity()

    elif name =='Log':
        trans = transform.Log()
        trans.params = 1.

    elif name =='Reciprocal':
        trans = transform.Reciprocal()
        trans.params = 1.

    else:
        expected = ['Identity', 'Log', 'Reciprocal']
        raise ValueError('Expected transform in {0}, got {1}'.format(\
            expected, name))

    return trans


def __check_ensemble_data(obs, ens):
    ''' Check dimensions of obs and ens data '''

    obs = np.atleast_1d(obs).astype(np.float64)
    ens = np.atleast_2d(ens).astype(np.float64)

    nforc = obs.shape[0]
    nens = ens.shape[1]
    if ens.shape[0]!=nforc:
        raise ValueError('Expected ens with first dim equal to {0}, got {1}'.format( \
            nforc, ens.shape[0]))

    return obs, ens, nforc, nens


def pit(obs, ens, cst=0.3):
    """
    Compute probability integral transformed (PIT) values
    for a single forecast

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array
    cst : float
        Constant used to compute plotting positions

    Returns
    -----------
    pits : numpy.ndarray
        PIT value
    """
    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Initialise outputs
    pits = np.zeros(nforc)
    unif_dist = sutils.ppos(nens, cst)

    # Loop through forecasts
    for i in range(nforc):
        rnd = np.random.uniform(0, EPS, size=nens)
        ys_sort = np.sort(ens[i, :]+rnd)

        # Compute PIT
        prob_obs = 0.0
        if obs[i]>= ys_sort[-1]:
            prob_obs = 1.0

        elif obs[i]>= ys_sort[0]:
            prob_obs = np.interp(obs[i], ys_sort, unif_dist)

        pits[i] = prob_obs

    return pits


def crps(obs, ens):
    ''' Compute the CRPS decomposition from Hersbach (2000)

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array

    Returns
    -----------
    decompos : pandas.Series
        CRPS decomposition as per Equations 35 to 39 in Hersbach, 2000
    table : pandas.DataFrame
        Decomposition table, as per
        - Equation 29 for a and b (alpha and beta)
        - Equation 30 for g
        - Equation 36 for reliability
    '''
    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # set weights to zero and switch off use of weights
    weights = np.zeros(nforc)
    use_weights = 0

    # run C code via cython
    table = np.zeros((nens+1, 7))
    decompos = np.zeros(5)
    is_sorted = 0
    ierr = c_hydrodiy_stat.crps(use_weights, is_sorted, obs, ens,
                    weights, table, decompos)

    table = pd.DataFrame(table)
    table.columns = ['freq', 'a', 'b', 'g', 'rank',
                'reliability', 'crps_potential']

    decompos= pd.Series(decompos, \
        index = ['crps', 'reliability', 'resolution', \
            'uncertainty', 'potential'])

    return decompos, table


def cramer_von_mises_test(data):
    ''' Perform the Cramer-Von Mises test using a uniform
    distribution as a null hypothesis. The pvalue are computed
    from table prepared with the R package goftest (function pCvM)

    Parameters
    -----------
    data : numpy.ndarray
        1d data vector

    Returns
    -----------
    pvalue : float
        CV test pvalue
    cvstat : float
        CV test statistic
    '''

    # Compute Cramer-Von Mises statistic
    nsample = data.shape[0]
    unif = (2*np.arange(1, nsample+1)-1).astype(float)/2/nsample
    cvstat = 1./12/nsample + np.sum((unif-np.sort(data))**2)

    # Find closest sample population
    idx = np.argmin(np.abs(nsample-CVNSAMPLE))
    cdf = CVTABLE[:, idx]

    # Interpolate pvalue
    pvalue = np.interp(cvstat, CVQQ, cdf)

    return cvstat, pvalue


def alpha(obs, ens, cst=0.3):
    ''' Score computing the Pvalue of the Kolmogorov-Smirnov test
    and Cramer-Von Mises test

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        simulated data, [n,p] array
    cst : float
        Constant used in the computation of the plotting position

    Returns
    -----------
    kstat : float
        KS test statistic
    kpvalue : float
        KS test pvalue
    cstat : float
        Cramer Von Mises test statistic
    cpvalue : float
        Cramer Von Mises test pvalue
    '''
    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Compute pit
    pits = pit(obs, ens)

    # KS test
    kstat, kpvalue = kstest(pits, 'uniform')

    # Cramer Von-Mises test
    cstat, cpvalue = cramer_von_mises_test(pits)

    return kstat, kpvalue, cstat, cpvalue


def iqr(ens, ref, coverage=50.):
    ''' Compute the interquantile range skill score (iqr)

    Parameters
    -----------
    ens : numpy.ndarray
        simulated data, [n,p] array
    ref : numpy.ndarray
        climatology data, [n,p] array
    coverage : float
        Interval coverage. For example, if coverage=50,
        the score will be computed between 25% and 75% percentiles

    Returns
    -----------
    skill : float
        IQR skill score computed as (see below)
        (clim-score)/(clim+score)
    score : float
        IQR score
    clim : float
        IQR score of climatology
    '''

    # Check data
    ens = np.atleast_2d(ens)
    ref = np.atleast_2d(ref)
    nforc, nens = ens.shape

    if ens.shape != ref.shape:
        raise ValueError('Expected ref with dim equal to {0}, got{1}'.format( \
            ens.shape, ref.shape))

    # Initialise
    iqr = np.zeros((nforc, 3))
    iqr_clim = np.zeros((nforc, 3))

    # Coverage percentiles
    perc =[coverage/2, 100.-coverage/2]

    # Loop through forecasts
    for i in range(nforc):
        iqr_clim[i, :2] = np.percentile(ref[i,:], perc)
        iqr_clim[i, 2] = iqr_clim[i,1]-iqr_clim[i,0]

        iqr[i, :2] = np.percentile(ens[i,:], perc)
        iqr[i, 2] = iqr[i,1]-iqr[i,0]

    skill = 100*np.mean((iqr_clim[:, 2]-iqr[:, 2])/(iqr_clim[:, 2]+iqr[:, 2]))
    score = np.mean(iqr[:, 2])
    clim = np.mean(iqr_clim[:, 2])

    return skill, score, clim


def bias(obs, sim, transform='Identity'):
    ''' Compute simulation bias

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    transform : str
        Name of data transformation: Identity, Log or Reciprocal

    Returns
    -----------
    bias : float
        Simulation bias
    '''
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape != sim.shape:
        raise ValueError('Expected sim with dim equal to {0}, got{1}'.format( \
            obs.shape, sim.shape))

    # Transform
    trans = get_transform(transform)
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Compute
    idx = pd.notnull(tobs) & pd.notnull(tsim)
    mo = np.mean(tobs[idx])
    ms = np.mean(tsim[idx])
    bias_value = (ms-mo)/mo

    return bias_value


def nse(obs, sim, transform='Identity'):
    ''' Compute Nash-Sucliffe efficiency

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    transform : str
        Name of data transformation: Identity, Log or Reciprocal

    Returns
    -----------
    nse_value : float
        Nash-Sutcliffe efficiency
    '''
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape != sim.shape:
        raise ValueError('Expected sim with dim equal to {0}, got{1}'.format( \
            obs.shape, sim.shape))

    # Transform
    trans = get_transform(transform)
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Compute
    idx = pd.notnull(tobs) & pd.notnull(tsim)
    err1 = np.sum((tsim[idx]-tobs[idx])**2)

    mo = np.mean(tobs[idx])
    err2 = np.sum((mo-tobs[idx])**2)
    nse_value = 1-err1/err2

    return nse_value



