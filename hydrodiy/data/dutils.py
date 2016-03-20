import re
import math

from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

import numpy as np
import pandas as pd


def normaliseid(id):
    ''' Normalise station id by removing trailing letters,
    underscores and leading zeros

    Parameters
    -----------
    params : str
        Station id

    Returns
    -----------
    idn : str
        Normalise id

    Example
    -----------
    Normalise a complex id
    >>> dutils.normaliseid('00a04567B.100')
    'A04567'

    '''

    idn = '%s' % id

    if re.search('[0-9]', idn):
        idn = re.sub('^0*|[A-Z]*$', '', idn)

    # Remove spaces
    idn = re.sub(' ', '', idn)

    # Remove delimiters
    idn = re.sub('\\(|\\)|-', '_', idn)

    # Remove multiple underscores
    idn = re.sub('_+', '_', idn)

    # Remove character after a dot and first and last underscores
    idn = re.sub('\\..*$|\\_*$|^\\_*', '', idn)

    # Get upper case
    idn = idn.upper()

    return idn


def aggmonths(ts, nmonths=3, ngapmax=6, ngapcontmax=3):
    ''' Convert time series to aggregated monthly time steps

    Parameters
    -----------
    ts : pandas.core.series.Series
        Input time series
    nmonths : int
        Number of months used for aggregation
    ngapmax : int
        Maximum number of missing values within a month
    ngapcontmax : int
        Maximum number of continuous missing values within a month

    Returns
    -----------
    out : pandas.core.series.Series
        Aggregated time series

    Example
    -----------
    >>> import pandas as pd
    >>> from hydata import dutils
    >>> idx = pd.date_range('1980-01-01', '1980-12-01', freq='MS')
    >>> ts = pd.Series(range(12), index=idx)
    >>> dutils.to_seasonal(ts)
    1980-01-01     3
    1980-01-02     6
    1980-01-03     9
    1980-01-04    12
    1980-01-05    15
    1980-01-06    18
    1980-01-07    21
    1980-01-08    24
    1980-01-09    27
    1980-01-10    30
    1980-01-11   NaN
    1980-01-12   NaN
    Freq: MS, dtype: float64
    '''

    # Resample to monthly
    tsmm = ts.groupby(ts.index.month).mean()

    def _sum(x):
        # Count gaps
        ngap = np.sum(pd.isnull(x))

        # Count continuous gaps
        ngapcont = 0
        if ngap > 0 :
            y = []
            for i in range(ngapcontmax+1):
                y.append(x.shift(i))
            y = pd.DataFrame(y).isnull()

            ys = y.sum(axis=0)
            ngapcont = ys[ngapcontmax:].max()

        # Return sum
        if (ngap > ngapmax) | (ngapcont > ngapcontmax):
            return np.nan

        elif ngap == 0:
            return x.sum()

        else:
            mv = tsmm[x.index.month[0]]
            xx = x.fillna(mv)
            return xx.sum()

    tsm = ts.resample('MS', how=_sum)

    # Shift series
    tss = []
    for s in range(0, nmonths):
        tss.append(tsm.shift(-s))

    tss = pd.DataFrame(tss)
    out = tss.sum(axis=0, skipna=False)

    return out


def atmpressure(altitude):
    ''' Compute mean atmospheric pressure
        See http://en.wikipedia.org/wiki/Atmospheric_pressure

    Parameters
    -----------
    altitude : numpy.ndarray
        Altitude from mean sea level (m)

    Returns
    -----------
    P : numpy.ndarray
        Atmospheric pressure

    Example
    -----------
    >>> import numpy as np
    >>> from hydrodiy.hydata import dutils
    >>> alt = np.linspace(0, 1000, 10)
    >>> dutils.to_seasonal(alt)
    1980-01-01     3
    1980-01-02     6
    1980-01-03     9
    1980-01-04    12
    1980-01-05    15
    1980-01-06    18
    1980-01-07    21

    '''

    g = 9.80665 # m/s^2 - Gravity acceleration
    M = 0.0289644 # kg/mol - Molar mass of dry air
    R = 8.31447 # j/mol/K - Universal gas constant
    T0 = 288.15 # K - Sea level standard temp
    P0 = 101325 # Pa - Sea level standard atmospheric pressure

    P = P0 * np.exp(-g*M/R/T0 * altitude)

    return P


