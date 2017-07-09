
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
    cst = min(0.5, cst)
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Compute pits
    dobs = np.random.uniform(-EPS, EPS, size=nforc)
    pits = (ens-(obs+dobs)[:, None]<0).astype(int)
    pits = (np.sum(pits, 1)+0.5-cst)/(1.-cst+nens)

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


def alpha(obs, ens, cst=0.3, type='CV'):
    ''' Score computing the Pvalue of the Cramer Von-Mises test (CV) or
    Kolmogorov-Smirnov test (KS)

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        simulated data, [n,p] array
    cst : float
        Constant used in the computation of the plotting position
    type : str
        Type of alpha score. CV is Cramer Von-Mises, KS is Kolmogorov-Smirnov

    Returns
    -----------
    stat : float
        Test statistic (low values mean that the test is passed)
    pvalue : float
        Test pvalue (values close to one mean that the test is passed)
    '''
    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Compute pit
    pits = pit(obs, ens)

    if type == 'KS':
        # KS test
        stat, pvalue = kstest(pits, 'uniform')
    elif type == 'CV':
        # Cramer Von-Mises test
        stat, pvalue = cramer_von_mises_test(pits)
    else:
        raise ValueError('Expected test type in [CV/KS], got {0}'.format(type))

    return stat, pvalue


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
        IQR skill score computed as
        1/n Sum (clim[i]-score[i])/(clim[i]+score[i])

        Interpretation of this scores is:
        - a value close to 100 indicates perfect IQR, or that the forecast is
          close to a deterministic ensemble
        - a value close to 0 indicates same IQR than climatology
        - a value close to -100 indicates that climatologyis close to a deterministic
          ensemble

    score : float
        IQR score

    clim : float
        IQR score of climatology

    ratio : float
        IQR ratio computed as
        1/n Sum score[i]/clim[i]

        Interpretation of this scores is:
        - a value close to 0 indicates perfect IQR, or that the forecast is
          close to a deterministic ensemble
        - a value close to 1 indicates same IQR than climatology
        - a value close to +infinity indicates that climatologyis close to a deterministic
          ensemble
    '''

    # Check data
    ens = np.atleast_2d(ens)
    ref = np.atleast_2d(ref)
    nforc, _ = ens.shape

    # Initialise
    iqr = np.zeros((nforc, 3))
    iqr_clim = np.zeros((nforc, 3))

    # Coverage percentiles
    perc =[coverage/2, 100.-coverage/2]

    # Loop through forecasts
    for i in range(nforc):
        iqr_clim[i, :2] = np.nanpercentile(ref[i,:], perc)
        iqr_clim[i, 2] = iqr_clim[i,1]-iqr_clim[i,0]

        iqr[i, :2] = np.nanpercentile(ens[i,:], perc)
        iqr[i, 2] = iqr[i,1]-iqr[i,0]

    skill = 100*np.mean((iqr_clim[:, 2]-iqr[:, 2])/(iqr_clim[:, 2]+iqr[:, 2]))
    ratio = np.mean(iqr[:, 2]/iqr_clim[:, 2])
    score = np.mean(iqr[:, 2])
    clim = np.mean(iqr_clim[:, 2])

    return skill, score, clim, ratio


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
    ''' Compute Nash-Sucliffe efficiency. If the sim data is an
    ensemble, uses the NSE probabilistic formulation introduced by
    (find the reference)

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
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

    if obs.shape[0] != sim.shape[0]:
        raise ValueError('Expected sim with dim equal to {0}, got {1}'.format( \
            obs.shape[0], sim.shape[0]))

    # Check if sim is an ensemble
    ens = False
    if sim.ndim>1:
        if sim.shape[1]>1:
            ens = True

    # Transform
    trans = get_transform(transform)
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Select non null obs data
    idx = pd.notnull(tobs)

    if ens:
        # Ensures that at least 1 simulated value is not null
        idx = idx & ~np.all(pd.isnull(tsim), axis=1)

        # Ensemble means
        ms = np.mean(tsim[idx], 1)

        # Ensemble variance
        vs = np.var(tsim[idx], 1)

        # Compute probabilistic SSE
        errs_accur = np.sum((ms-tobs[idx])**2)
        errs_sharp = np.sum(vs)

    else:
        # Compute regular SSE
        idx = idx & pd.notnull(tsim)
        errs_accur = np.sum((tsim[idx]-tobs[idx])**2)
        errs_sharp = 0.

    mo = np.mean(tobs[idx])
    erro = np.sum((mo-tobs[idx])**2)

    value = 1-(errs_sharp+errs_accur)/erro
    accur = 1-errs_accur/erro
    sharp = 1-errs_sharp/erro

    return value, accur, sharp


def dscore(obs, sim):
    ''' Compute the discrimination score (D score) for continuous
    forecasts as per

    Weigel, Andreas P., and Simon J. Mason.
    "The generalized discrimination score for ensemble forecasts."
    Monthly Weather Review 139.9 (2011): 3069-3074.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array

    Returns
    -----------
    D : float
        D score value
    '''
    # Check data
    obs = np.atleast_1d(obs).astype(np.float64)
    sim = np.atleast_1d(sim).astype(np.float64)

    if sim.ndim != 2:
        raise ValueError('Expected sim of dimension 2, '+\
            'got {0}'.format(sim.shape))

    nval, nens = sim.shape

    if nens == 1:
        # Compute ensemble rank for deterministic forecasts
        franks = np.argsort(np.argsort(sim[:, 0]))
    else:
        # initialise data
        fmat = np.zeros((nval, nval), dtype=np.float64)
        franks = np.zeros(nval, dtype=np.float64)

        # Compute ensemble rank for ensemble forecasts
        c_hydrodiy_stat.ensrank(sim, fmat, franks)

    # Compute obs rank
    oranks = np.argsort(np.argsort(obs))

    # Compute rank correlation
    D = (np.corrcoef(oranks, franks)[0, 1]+1)/2

    return D

