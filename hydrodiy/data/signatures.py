''' Compute signature from times series '''

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

from hydrodiy.data.qualitycontrol import ismisscens
from hydrodiy.data.dutils import lag, flathomogen
from hydrodiy.stat.sutils import ppos
from hydrodiy.stat.metrics import corr, nse
from hydrodiy.stat.transform import Identity, Log

# Try to import C code
HAS_C_MODULES = True
try:
    import c_hydrodiy_data
except ImportError:
    HAS_C_MODULES = False

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
    # Check C modules are available
    if not HAS_C_MODULES:
        raise ValueError('Compiled C modules are not available, '+\
                'please run python setup.py build')

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


def fdcslope(x, q1=90, q2=100, cst=0.375, trans=Identity()):
    ''' Slope of flow duration curve as per
    Yilmaz, Koray K., Hoshin V. Gupta, and Thorsten Wagener.
    A process-based diagnostic approach to model
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
    trans : hydrodiy.stat.transform.Transform
        Data transform object.

    Returns
    -----------
    slp : float
        Slope of flow duration curve (same unit than x)
    qq : numpy.ndarray
        The two quantiles corresponding to q1 and q2
    '''
    # Check data
    icens = ismisscens(x)
    iok = icens > 0
    nok = np.sum(iok)
    if nok == 0:
        raise ValueError('No valid data')

    if q2 < q1 + 1./nok:
        raise ValueError('Expected q2 > q1, got q1={0} and q2={1}'.format(\
                    q1, q2))

    if np.nanstd(x) < EPS:
        return np.nan, (np.nan, np.nan)

    # Compute percentiles
    xok = np.sort(x[iok])
    qq = np.percentile(xok, [q1, q2])
    idxqq = (xok>=qq[0]) & (xok<=qq[1])
    nqq = np.sum(idxqq)
    if nqq == 0:
        raise ValueError('No data in range Q{0}-Q{1}'.format(q1, q2))

    # Transform
    txr = trans.forward(np.sort(xok[idxqq]))

    # Compute frequencies
    ff = ppos(len(xok), cst=cst)[idxqq]

    # Compute slope
    M = np.column_stack([np.ones(nqq), ff])
    theta, _ , _, _ = np.linalg.lstsq(M, txr)

    return theta[1], qq


def goue(aggindex, values, trans=Identity()):
    ''' GOUE index (Nash sutcliffe of flat disaggregated daily vs daily)
    as per
    Ficchi, Andrew, Charles Perrin, and Vazken Andreassian
     Impact of temporal resolution of inputs on hydrological
     model performance: An analysis based on 2400 flood events.
     Journal of Hydrology 538 (2016): 454-470.

    Parameters
    -----------
    aggindex : numpy.ndarray
        Aggregation index (e.g. month in the form 199501 for
        Jan 1995). Index should be in increasing order.
        (see also hydrodiy.data.dutils.aggregate)
    values : numpy.ndarray
        Values to be homogeneise
    trans : hydrodiy.stat.transform.Transform
        Data transform object

    Returns
    -----------
    goue_value : float
        GOUE index value
    '''
    # Get flat homogeneised series
    values_flat = flathomogen(aggindex, values)

    # Compute goue
    goue_value = nse(values, values_flat, trans=trans)

    return goue_value


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
    icens = ismisscens(x)

    # Shift data
    xshifted = lag(x, 1)

    # Correlation
    return corr(x, xshifted, censor=censor, type=type)

