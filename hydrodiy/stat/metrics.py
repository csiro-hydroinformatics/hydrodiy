import re, os, math
import pkg_resources

import numpy as np
import pandas as pd

from scipy.stats import kstest, percentileofscore

import warnings

from hydrodiy.stat import transform
from hydrodiy.stat import sutils
from hydrodiy.io import csv

import c_hydrodiy_stat

EPS = 1e-10

# Reads Cramer-Von Mises table
CVPATH = pkg_resources.resource_filename(__name__, \
                    os.path.join('data', 'cramer_von_mises_test_pvalues.zip'))
CVM_TABLE, _ = csv.read_csv(CVPATH, index_col=0)
CVM_NSAMPLE  = CVM_TABLE.columns.values.astype(int)
CVM_QQ = CVM_TABLE.index.values
CVM_TABLE = CVM_TABLE.values


def __check_ensemble_data(obs, ens):
    ''' Check dimensions of obs and ens data '''

    obs = np.atleast_1d(obs).astype(np.float64)
    if obs.ndim > 1:
        obs = obs.squeeze()
    if obs.ndim > 1:
        raise ValueError('obs is not 1D')

    ens = np.atleast_2d(ens).astype(np.float64)

    nforc = obs.shape[0]
    nens = ens.shape[1]
    if ens.shape[0]!=nforc:
        raise ValueError('Expected ens with first dim equal to'+\
            ' {0}, got {1}'.format( \
            nforc, ens.shape[0]))

    return obs, ens, nforc, nens


def pit(obs, ens, random=False, cst=0.3, kind='rank', censor=0.):
    """
    Compute probability integral transformed (PIT) values
    for ensemble forecasts.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array
    random : bool
        Randomize forecast and obs to generate pseudo-pit
    cst : float
        Constant used to compute plotting positions if random is True
    kind : str
        Argument passed to percentileofscore function from scipy
    censor : float
        Censoring threshold to compute sudo pit (i.e. when obs <= censor
        and ens has members <= censor)

    Returns
    -----------
    pits : numpy.ndarray
        PIT value
    is_sudo : numpy.ndarray
         Tells if the pit is a sudo value
    """
    # Check data
    cst = min(0.5, cst)
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Check sudo pits
    is_sudo = np.zeros(nforc).astype(bool)
    idx = (obs < censor+EPS) & (np.sum(ens < censor + EPS, axis=1) > 0)
    is_sudo[idx] = True

    # Compute pits
    if random:
        dobs = np.random.uniform(-EPS, EPS, size=nforc)
        dens = np.random.uniform(-EPS, EPS, size=(nforc, nens))
        pits = (ens+dens-(obs+dobs)[:, None]<0).astype(int)
        pits = (np.sum(pits, 1)+0.5-cst)/(1.-cst+nens)
    else:
        pits = np.array([percentileofscore(ensval, obsval, kind)/100. \
                    for ensval, obsval in zip(ens, obs)])

    return pits, is_sudo


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
    weights = np.zeros(nforc, dtype=np.float64)
    use_weights = 0

    # run C code via cython
    table = np.zeros((nens+1, 7), dtype=np.float64)
    decompos = np.zeros(5, dtype=np.float64)
    is_sorted = 0
    ierr = c_hydrodiy_stat.crps(use_weights, is_sorted, obs, ens,
                    weights, table, decompos)

    if ierr!=0:
        raise ValueError('c_crps returns %d'%ierr)

    table = pd.DataFrame(table)
    table.columns = ['freq', 'a', 'b', 'g', 'rank',
                'reliability', 'crps_potential']

    decompos= pd.Series(decompos, \
        index = ['crps', 'reliability', 'resolution', \
            'uncertainty', 'potential'])

    return decompos, table


def anderson_darling_test(unifdata):
    ''' Compute the Anderson Darling (AD) test statistic for a
    uniformly distributed variable and its pvalue using the code
    provided by Marsaglia and Marsaglia (2004):
    Marsaglia, G., & Marsaglia, J. (2004).
    Evaluating the Anderson-Darling Distribution. Journal of Statistical
    Software, 9(2), 1 - 5. doi:http://dx.doi.org/10.18637/jss.v009.i02

    Parameters
    -----------
    unifdata : numpy.ndarray
        1d data vector in [0, 1]

    Returns
    -----------
    pvalue : float
        AD test pvalue
    adstat : float
        AD test statistic
    '''

    # Check data
    unifdata = np.atleast_1d(unifdata).astype(np.float64)

    # set function outputs
    outputs = np.zeros(2, dtype=np.float64)

    # run C code via cython
    ierr = c_hydrodiy_stat.ad_test(unifdata, outputs)

    if ierr!=0:
        raise ValueError('ad_test returns %d'%ierr)

    adstat = outputs[0]
    pvalue = outputs[1]

    return adstat, pvalue


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
    idx = np.argmin(np.abs(nsample-CVM_NSAMPLE))
    cdf = CVM_TABLE[:, idx]

    # Interpolate pvalue
    pvalue = np.interp(cvstat, CVM_QQ, cdf)

    return cvstat, pvalue


def alpha(obs, ens, cst=0.3, type='CV', sudo_perc_threshold=5):
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
        Type of alpha score. CV is Cramer Von-Mises, KS is Kolmogorov-Smirnov,
        AD is Anderson Darling.
    sudo_perc_threshold : float
        Percentage threshold for warning about too many sudo pits

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
    pits, is_sudo = pit(obs, ens)

    # Warning if too much sudo pits
    if np.sum(is_sudo) > nforc*float(sudo_perc_threshold)/100:
        warnings.warn(('More than {0}% sudo pits in pits'+\
                        ' series').format(sudo_perc_threshold))

    if type == 'KS':
        # KS test
        stat, pvalue = kstest(pits, 'uniform')
    elif type == 'CV':
        # Cramer Von-Mises test
        stat, pvalue = cramer_von_mises_test(pits)
    elif type == 'AD':
        # Anderson Darlin test
        stat, pvalue = anderson_darling_test(pits)
    else:
        raise ValueError('Expected test type in [CV/KS/AD],'+\
                ' got {0}'.format(type))

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
        - a value close to -100 indicates that climatologyis is close
          to a deterministic ensemble

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
        - a value close to +infinity indicates that climatology is close
          to a deterministic ensemble
    '''

    # Check data
    ens = np.atleast_2d(ens)
    ref = np.atleast_2d(ref)
    nforc, _ = ens.shape

    if ref.shape[0] != nforc:
        raise ValueError(('Expected clim to have {0} forecasts, '+\
                        'got {1}').format(nforc, ens.shape[0]))

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


def bias(obs, sim, trans=transform.Identity()):
    ''' Compute simulation bias

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    transform : hydrodiy.stat.transform.Transform
        Data transforma object

    Returns
    -----------
    bias : float
        Simulation bias
    '''
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape != sim.shape:
        raise ValueError('Expected sim with dim equal '+\
            'to {0}, got{1}'.format( \
            obs.shape, sim.shape))

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Compute
    idx = pd.notnull(tobs) & pd.notnull(tsim)
    meano = np.mean(tobs[idx])
    if abs(meano) < EPS:
        warnings.warn(('Mean value of obs is close to '+\
                'zero ({0:3.3e}), returning nan').format(\
                    meano))
        return np.nan

    means = np.mean(tsim[idx])
    bias_value = (means-meano)/meano

    return bias_value


def nse(obs, sim, trans=transform.Identity()):
    ''' Compute Nash-Sucliffe efficiency.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
    trans : hydrodiy.stat.transform.Transform
        Data transform object

    Returns
    -----------
    value : float
        Nash-Sutcliffe efficiency (N)

    '''
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape[0] != sim.shape[0]:
        raise ValueError('Expected sim with dim equal '+\
            'to {0}, got {1}'.format( \
            obs.shape[0], sim.shape[0]))

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Select non null obs data
    idx = pd.notnull(tobs)

    # SSE
    idx = idx & pd.notnull(tsim)
    errs = np.sum((tsim[idx]-tobs[idx])**2)

    mo = np.mean(tobs[idx])
    erro = np.sum((mo-tobs[idx])**2)

    value = 1-errs/erro

    return value


def dscore(obs, sim, eps=1e-6):
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
    eps : float
        Tolerance to detect ties

    Returns
    -----------
    D : float
        D score value:
        * D=0.0 means that the model as an inverse discrimination (i.e.
                forecasting high when obs is low)
        * D=0.5 means that the model is not discriminating
        * D=1.0 means that the model is perfectly discriminating

    '''
    # Check data
    obs = np.atleast_1d(obs).astype(np.float64)
    sim = np.atleast_2d(sim).astype(np.float64)
    eps = np.float64(eps)

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
        c_hydrodiy_stat.ensrank(eps, sim, fmat, franks)

    # Compute obs rank
    oranks = np.argsort(np.argsort(obs))

    # Compute rank correlation
    D = (np.corrcoef(oranks, franks)[0, 1]+1)/2

    return D


def kge(obs, sim, trans=transform.Identity()):
    ''' Compute Kling-Gupta efficiency.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
    trans : hydrodiy.stat.transform.Transform
        Data transform object

    Returns
    -----------
    value : float
        Nash-Sutcliffe efficiency (N)

    '''
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape[0] != sim.shape[0]:
        raise ValueError('KGE - Expected sim with dim equal '+\
            'to {0}, got {1}'.format( \
            obs.shape[0], sim.shape[0]))

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    # Select non null obs data
    idx = pd.notnull(tobs) & pd.notnull(tsim)
    tobs = tobs[idx]
    tsim = tsim[idx]

    # Means
    meano = np.mean(tobs)
    if abs(meano) < EPS:
        warnings.warn(('KGE - Mean value of obs is close to '+\
                'zero ({0:3.3e}), returning nan').format(\
                    meano))
        return np.nan

    means = np.mean(tsim)

    # Standard deviations
    stdo = np.std(tobs)
    stds = np.std(tsim)

    if abs(stdo) < EPS:
        warnings.warn(('KGE - Standard dev of obs is close to '+\
                'zero ({0:3.3e}), returning nan').format(\
                    stdo))
        return np.nan

    # Correlation
    if abs(stds) > EPS:
        corr = np.corrcoef(tobs, tsim)[0, 1]
    else:
        warnings.warn(('KGE - Standard dev of sim is close to '+\
                'zero ({0:3.3e}), cannot compute correlation, '+\
                'returning nan').format(stds))
        return np.nan

    # KGE
    value = 1-math.sqrt((1-means/meano)**2+(1-stds/stdo)**2+(1-corr)**2)

    return value


