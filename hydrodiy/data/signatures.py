''' Compute signature from times series '''

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

from hydrodiy.data.qualitycontrol import check1d
from hydrodiy.stat.sutils import ppos
from hydrodiy.stat.metrics import corr

EPS = 1e-10


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
    x, idxok, nok = check1d(x)
    if q2 < q1 + 1:
        raise ValueError('Expected q2 > q1, got q1={0} and q2={1}'.format(\
                    q1, q2))

    # Compute percentiles
    xok = x[idxok]
    qq = np.percentile(xok, [q1, q2])
    idx = (xok>=qq[0]) & (xok<=qq[1])
    nqq = np.sum(idx)
    if nqq == 0:
        raise ValueError('No data selected in range')

    # Select data and sort
    xr = np.sort(xok)

    # Compute frequencies
    ff = ppos(nok, cst=cst)

    # Check data is not constant
    if np.std(x) < EPS:
        return np.nan

    # Compute slope
    M = np.column_stack([np.ones(nqq), xr[idx]])
    theta, _ , _, _ = np.linalg.lstsq(M, ff[idx])
    import pdb; pdb.set_trace()

    return theta[1], qq

def goue(x):
    ''' GOUE index (Nash sutcliffe of flat disaggregated daily vs daily)
    as per
    Ficchì, Andrea, Charles Perrin, and Vazken Andréassian.
     "Impact of temporal resolution of inputs on hydrological
     model performance: An analysis based on 2400 flood events."
     Journal of Hydrology 538 (2016): 454-470.

    Parameters
    -----------
    x : pandas.Series
        Input daily time series

    Returns
    -----------
    goue_value : float
        GOUE index value
    '''
    # Check data
    times = x.index
    x, idxok, nok = check1d(x)



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

