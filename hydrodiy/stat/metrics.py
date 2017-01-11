
import re

import numpy as np
import pandas as pd

from scipy.special import kolmogorov
from scipy.stats import kendalltau

from hydrodiy.stat import transform
from hydrodiy.stat import sutils

import c_hydrodiy_stat

EPS = 1e-10

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
        raise ValueError('Transform {0} not recognised'.format(name))

    return trans


def pit(obs, sim, cst=0.3):
    """
    Compute probability integral transformed (PIT) values
    for a single forecast

    Parameters
    -----------
    obs : numpy.ndarray
        observed  data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    cst : float
        Constant used to compute plotting positions

    Returns
    -----------
    prob_obs : float
        PIT value
    """

    rnd = np.random.uniform(0, EPS, size=sim.shape)
    ys_sort = np.sort(sim+rnd)
    prob_obs = 0.0

    if obs>= ys_sort[-1]:
        prob_obs = 1.0

    elif obs>= ys_sort[0]:
        unif_dist = sutils.ppos(len(ys_sort), cst)
        prob_obs = np.interp(obs, ys_sort, unif_dist)

    return prob_obs


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
    obs = np.atleast_1d(obs)
    ens = np.atleast_2d(ens)

    nforc = obs.shape[0]
    nens = ens.shape[1]
    if ens.shape[0]!=nforc:
        raise ValueError('Expected ens with first dim equal to {0}, got{1}'.format( \
            nforc, ens.shape[0]))

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


def alpha(obs, ens, cst=0.3):
    ''' Score computing the Pvalue of the Kolmogorov-Smirnov test

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
    pvalue : float
        KS test pvalue
    kstat : float
        KS test statistic
    '''

    # Check data
    obs = np.atleast_1d(obs)
    ens = np.atleast_2d(ens)

    nforc = obs.shape[0]
    nens = ens.shape[1]
    if ens.shape[0]!=nforc:
        raise ValueError('Expected ens with first dim equal to {0}, got{1}'.format( \
            nforc, ens.shape[0]))

    # Compute pit
    pt = [pit(o, s, has_ties=True, cst=cst)
                    for o,s in zip(obs, ens)]

    # Distance between cumulative distribution of pit and 1:1
    unif_dist = sutils.ppos(len(ys_sort), cst)
    distances = np.sort(pt)-unif_dist
    max_dist = np.max(np.abs(distances))

    # KS statistic
    kstat = np.sqrt(nval)*max_dist
    pvalue = 100*kolmogorov(np.sqrt(nval)*max_dist)

    return pvalue, kstat


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



