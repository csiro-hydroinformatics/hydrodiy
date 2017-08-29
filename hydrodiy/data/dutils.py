import re
import math

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta
import calendar

import numpy as np
import pandas as pd

from pandas.tseries.offsets import DateOffset
from numpy.polynomial import polynomial as poly

import c_hydrodiy_data


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
    altitude : float
        Altitude from mean sea level (m)

    Returns
    -----------
    P : float
        Atmospheric pressure
    '''

    g = 9.80665 # m/s^2 - Gravity acceleration
    M = 0.0289644 # kg/mol - Molar mass of dry air
    R = 8.31447 # j/mol/K - Universal gas constant
    T0 = 288.15 # K - Sea level standard temp
    P0 = 101325 # Pa - Sea level standard atmospheric pressure

    P = P0 * np.exp(-g*M/R/T0 * altitude)

    return P


def aggregate(aggindex, inputs, oper=0, maxnan=0):
    ''' Fast aggregation of inputs based on aggregation indices
        This is an equivalent of pandas.Series.resample method,
        but much faster.

    Parameters
    -----------
    aggindex : numpy.ndarray
        Aggregation index (e.g. month in the form 199501 for
        Jan 1995). Index should be in increasing order.
    inputs : numpy.ndarray
        Inputs data to be aggregated
    oper : int
        Aggregation operator:
        0 = sum
        1 = mean
    maxnan : int
        Maximum number of nan in inputs for each
        aggregation index

    Returns
    -----------
    outputs : numpy.ndarray
        Aggregated data
    '''

    # Allocate arrays
    oper = np.int32(oper)
    maxnan = np.int32(maxnan)
    aggindex = np.array(aggindex).astype(np.int32)
    inputs = inputs.astype(np.float64)
    outputs = 0.*inputs
    iend = np.array([0]).astype(np.int32)

    # Run C function
    ierr = c_hydrodiy_data.aggregate(oper, maxnan, aggindex, \
                inputs, outputs, iend)

    if ierr>0:
        raise ValueError('c_hydrodiy_data.aggregate returns {0}'.format(ierr))

    # Truncate the outputs to keep only the valid part of the vector
    outputs = outputs[:iend[0]]

    return outputs


def lag(data, lag):
    ''' Lag a numpy array and adds NaN at the beginning or end of the lagged data
        depending on the lag value. The lag is introduced using numpy.roll

        This is equivalent to pandas.DataFrame.shift function, but for Numpy
        arrays.

    Parameters
    -----------
    data : numpy.ndarray data
        Data series where linear interpolation is suspected
    lag : int
        Lag. If >0, lag the data forward. If<0 lag the data backward.

    Returns
    -----------
    lagged : numpy.ndarray
        Lagged vector

    '''
    # Check data
    data = np.atleast_1d(data)
    lag = int(lag)

    # Lagg data
    lagged = np.roll(data, lag, axis=0)
    if lag<0:
        lagged[lag:] = np.nan
    elif lag>0:
        lagged[:lag] = np.nan
    else:
        lagged = data.copy()

    return lagged


def monthly2daily(se, interpolation='flat', minthreshold=0.):
    ''' Convert monthly series to daily using interpolation.
    Takes care of the boundary effects.

    Parameters
    -----------
    sem : pandas.Series
        Monthly series
    interpolatoin : str
        Type of interpolation:
        * flat : flat disaggregation
        * cubic : cubic polynomial disaggregation preserving mass, continuity
                       and continuity of derivative
    minthreshold : float
        Minimum valid value

    Returns
    -----------
    se : pandas.Series
        Daily series
    '''
    # Set
    sec = se.copy()
    sec[pd.isnull(sec)] = minthreshold-1

    if interpolation == 'flat':
        # Add a fictive data after the last month
        # to allow for resample to work
        nexti = sec.index[-1] + delta(months=1)
        sec[nexti] = np.nan

        # Convert to daily
        sed = sec.resample('D').fillna(method='pad')
        sed /= sed.index.days_in_month
        sed[sed<minthreshold] = np.nan

        # Drop last values
        sed = sed.iloc[:-1]

    elif interpolation == 'cubic':
        # Prepare data
        start = sec.index[0]
        months = sec.index+DateOffset(months=1, days=-1)
        xc = np.concatenate([[0], (months-start).days + 1])
        xc2 = (xc[1:]+xc[:-1])/2
        dx = months.days_in_month
        y = sec.values.astype(float)

        # Interpolate linearly
        xx = np.arange(1, xc.max()).astype(float)
        yy = np.interp(xx, xc2, y)
        yyc = np.cumsum(yy)

        # Compute Derivatives
        u = y/dx
        dyc = np.concatenate([[u[0]], (u[1:]+u[:-1])/2, [u[-1]]])

        # Constraints for the computation of polynomial coefficients
        # 1. f(0) = 0
        # 2. f(1) = y[i]
        # 3. f'(0) = d0
        # 4. f'(1) = d1
        Mi = np.array([\
            [0., 1., 0.], \
            [3., -2., -1.], \
            [-2., 1., 1.]])

        ndays = np.array(months.days_in_month)
        const = np.array([y, dyc[:-1]*ndays, dyc[1:]*ndays])

        # Adjust constraint to ensure continuity of second derivative
        for i in range(const.shape[1]-1):
            F1 = const[0, i]/ndays[i]
            d0 = const[1, i]/ndays[i]
            d2 = const[2, i+1]/ndays[i+1]
            F2 = const[0, i+1]/ndays[i+1]
            d1 = (6*(F1+F2)-2*(d0+d2))/8
            const[2, i] = d1*ndays[i]
            const[1, i+1] = d1*ndays[i+1]

        # Compute polynomial coefficients
        coefs = np.insert(np.dot(Mi, const), 0, 0, axis=0).T

        # Evaluate polynomials
        xxt = np.repeat(np.arange(32)[None, :], len(y), 0).astype(float)
        xxt = xxt/ndays[:, None]
        xxt[xxt>1] = np.nan

        yyc = np.array([poly.polyval(t, c) for c, t in zip(coefs, xxt)])
        yy = np.diff(yyc, axis=1).ravel()
        yy = yy[~np.isnan(yy)]

        end = start + delta(days=len(yy))
        days = pd.date_range(start, end-delta(days=1))
        sed = pd.Series(yy, index=days)


    else:
        raise ValueError('Expected interpolation in [flat/cubic], got '+\
                    interpolation)

    return sed


