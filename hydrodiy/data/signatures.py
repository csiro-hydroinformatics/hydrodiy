''' Compute signature from times series '''

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

from hydrodiy.data.qualitycontrol import ismisscens
from hydrodiy.stat.sutils import ppos
from hydrodiy.stat.metrics import corr, nse
from hydrodiy.stat.transform import Identity

import c_hydrodiy_data

EPS = 1e-10


def eckhardt(flow, thresh=0.95, tau=20, BFI_max=0.8, timestep_type=1):
    ''' Compute the baseflow component based on Eckhardt algorithm
    Eckhardt K. (2005) How to construct recursive digital filters for
    baseflow separation. Hydrological processes 19:507-515.

    C code was translated from R code provided by
    Jose Manuel Tunqui Neira and Vazken Andreassian, IRSTEA

    Parameters
    -----------
    flow : numpy.array
        Streamflow data
    thresh : float
        Percentage from which the base flow should be considered
        as total flow
    tau : float
        Characteristic drainage timescale (hours)
    BFI_max : float
        See Eckhardt (2005)
    timestep_type : int
        Type of time step: 0=hourly, 1=daily

    Returns
    -----------
    bflow : numpy.array
        Baseflow time series

    Example
    -----------
    >>> import numpy as np
    >>> q = np.random.uniform(0, 100, size=1000)
    >>> signatures.baseflow(q)

    '''
    # run C code via cython
    thresh = np.float64(thresh)
    tau = np.float64(tau)
    BFI_max = np.float64(BFI_max)
    timestep_type = np.int32(timestep_type)

    flow = np.array(flow).astype(np.float64)
    bflow = np.zeros(len(flow), np.float64)

    ierr = c_hydrodiy_data.eckhardt(timestep_type, \
                thresh, tau, BFI_max, flow, bflow)

    if ierr!=0:
        raise ValueError('c_hydata.eckhardt returns %d'%ierr)

    return bflow


def fdcslope(x, q1=90, q2=100, cst=0.375):
    ''' Slope of flow duration curve as per
    Yilmaz, Koray K., Hoshin V. Gupta, and Thorsten Wagener.
    "A process‐based diagnostic approach to model
    evaluation: Application to the NWS distributed
    hydrologic model."Water Resources Research 44.9 (2008).

    Parameters
    -----------
    x : numpy.ndarray
        Input series
    q1 : float
        First percentile
    q2 : float
        Second percentile
    cst : float
        Constant used to compute plotting positions
        (see hydrodiy.stat.sutils.ppos)

    Returns
    -----------
    slp : float
        Slope of flow duration curve (same unit than x)
    qq : numpy.ndarray
        The two quantiles corresponding to q1 and q2
    '''
    # Check data
    icens = ismisscens(x)
    if q2 < q1 + 1:
        raise ValueError('Expected q2 > q1, got q1={0} and q2={1}'.format(\
                    q1, q2))

    # Compute percentiles
    xok = x[icens > 0]
    qq = np.percentile(xok, [q1, q2])
    idx = (xok>=qq[0]) & (xok<=qq[1])
    nqq = np.sum(idx)
    if nqq == 0:
        raise ValueError('No data selected in range')

    # Select data and sort
    xr = np.sort(xok)

    # Compute frequencies
    ff = ppos(len(xok), cst=cst)

    # Check data is not constant
    if np.std(x) < EPS:
        return np.nan

    # Compute slope
    M = np.column_stack([np.ones(nqq), xr[idx]])
    theta, _ , _, _ = np.linalg.lstsq(M, ff[idx])

    return theta[1], qq


def goue(daily, trans=Identity()):
    ''' GOUE index (Nash sutcliffe of flat disaggregated daily vs daily)
    as per
    Ficchì, Andrea, Charles Perrin, and Vazken Andréassian.
     "Impact of temporal resolution of inputs on hydrological
     model performance: An analysis based on 2400 flood events."
     Journal of Hydrology 538 (2016): 454-470.

    Parameters
    -----------
    daily : pandas.Series
        Input daily time series

    Returns
    -----------
    goue_value : float
        GOUE index value
    daily_flat : pandas.Series
        Flat-disaggregated daily time series
    monthly : pandas.Series
        Monthly time series
    '''
    # Monthly time series
    monthly = daily.copy().resample('MS').sum()

    # Upsample monthly to daily
    daily_flat = monthly.resample('D').pad()
    daily_flat /= daily_flat.index.daysinmonth

    # Fix end of series
    idx = daily.index.difference(daily_flat.index)
    if len(idx) > 0:
        se = pd.Series(np.nan, index=idx)
        daily_flat = daily_flat.append(se).sort_index()
        daily_flat = daily_flat.loc[daily.index]

    # Set nan
    daily_flat.loc[pd.isnull(daily)] = np.nan

    # Compute goue
    goue_value = nse(daily, daily_flat, trans=trans)

    return goue_value, daily_flat, monthly


def lag1corr(x, type='Pearson', censor=0.):
    ''' Compute the lag 1 autocorrelation with missing and censored data

    Parameters
    -----------
    x : numpy.ndarray
        Input series
    type : str
        Type of correlation. Can be Pearson, Spearman or censored.
        See hydrodiy.stat.metrics.corr
    censor : float
        Censoring threshold

    Returns
    -----------
    corr : float
        Lag 1 correlation
    '''
    # Check data
    x, idxok, nok = check1d(x)

    # Shift data
    xshifted = np.roll(x, 1)
    xshifted[0] = np.nan

    # Correlation
    return metrics.corr(x, xshifted, censor=censor, type=type)

