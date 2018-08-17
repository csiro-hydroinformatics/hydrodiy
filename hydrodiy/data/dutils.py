''' Utility functions to process data '''

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

from pandas.tseries.offsets import DateOffset
from numpy.polynomial import polynomial as poly

from hydrodiy import PYVERSION

import c_hydrodiy_data


def cast(x, y):
    ''' Cast y to the type of x.

        Useful to make sure that a function returns an output that has
        the same type than the input (e.g. to avoid mixing float with
        numpy.array 0d).


    Parameters
    -----------
    x : object
        First object
    y : object
        Second object
    '''
    # Check if x is an nxd numpy array (n>0)
    # then collect is type
    xdtype = None
    if hasattr(x, 'dtype'):
        # prevent the use of dtype for 0d array
        if x.ndim > 0:
            xdtype = x.dtype

    # Cast depending on the nature of x and y
    if xdtype is None:
        # x is a basic data type
        # this should work even if y is a
        # 1d or 0d numpy array

        # Except unsafe cast
        if isinstance(x, int) and isinstance(y, float):
            raise TypeError('Cannot cast value from float to int')

        ycast = type(x)(y)

    else:
        # x is a numpy array
        ycast = np.array(y).astype(xdtype, casting='safe')

    return ycast


def aggmonths(tseries, nmonths=3, ngapmax=6, ngapcontmax=3):
    ''' Convert time series to aggregated monthly time steps

    Parameters
    -----------
    tseries : pandas.core.series.Series
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

    # Aggregation function
    def sumfun(x):
        # Count gaps
        ngap = np.sum(pd.isnull(x))

        # Count continuous gaps
        ngapcont = 0
        if ngap > 0:
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
            fill = np.mean(x[pd.notnull(x)])
            xx = x.fillna(fill)
            return xx.sum()

    # Use Pandas 1 syntax, but can handle Pandas 0 too
    tsmr = tseries.resample('MS')
    if len(tsmr) == len(tseries):
        tsm = tsmr.apply(sumfun)
    else:
        tsm = tseries.resample('MS', how=sumfun)

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
    # Check inputs
    if len(aggindex) != len(inputs):
        raise ValueError('Expected inputs of length {0}, got {1}'.format(\
                len(aggindex), len(inputs)))

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

    if ierr > 0:
        raise ValueError('c_hydrodiy_data.aggregate'+\
                            ' returns {0}'.format(ierr))

    # Truncate the outputs to keep only the valid part of the vector
    outputs = outputs[:iend[0]]

    return outputs


def flatdisagg(aggindex, inputs, maxnan=0):
    ''' Compute a series of the same length than inputs with all values
    replaced by mean defined for each aggregation index.

    This function is used in hydrodiy.data.signatures.goue

    Parameters
    -----------
    aggindex : numpy.ndarray
        Aggregation index (e.g. month in the form 199501 for
        Jan 1995). Index should be in increasing order.
    inputs : numpy.ndarray
        Inputs data to be aggregated
    maxnan : int
        Maximum number of nan in inputs for each
        aggregation index

    Returns
    -----------
    outputs : numpy.ndarray
        Flat disaggregated data

    '''
    # Check inputs
    if len(aggindex) != len(inputs):
        raise ValueError('Expected inputs of length {0}, got {1}'.format(\
                len(aggindex), len(inputs)))

    # Allocate arrays
    maxnan = np.int32(maxnan)
    aggindex = np.array(aggindex).astype(np.int32)
    inputs = inputs.astype(np.float64)
    outputs = 0.*inputs

    # Run C function
    ierr = c_hydrodiy_data.flatdisagg(maxnan, aggindex, \
                inputs, outputs)

    if ierr > 0:
        raise ValueError('c_hydrodiy_data.flatdisagg'+\
                            ' returns {0}'.format(ierr))

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
    if lag < 0:
        lagged[lag:] = np.nan
    elif lag > 0:
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
        sed[sed < minthreshold] = np.nan

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
        xxt[xxt > 1] = np.nan

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


def var2h(se, maxgapsec=5*86400, display=False):
    ''' Convert a variable time step time series to hourly using
        linear interpolation and aggregation

    Parameters
    -----------
    se : pandas.Series
        Irregular time series
    maxgapsec : int
        Maximum number of seconds between two valid measurements
    display : bool
        Display progresses or not

    Returns
    -----------
    seh : pandas.Series
        Hourly series
   '''

    # Allocate arrays
    maxgapsec = np.int32(maxgapsec)
    display = np.int32(display)
    varvalues = se.values.astype(np.float64)
    varsec = (se.index.values.astype(np.int64)/1000000000).astype(np.int32)

    if maxgapsec < 3600:
        raise ValueError('Expected maxgapsec>=3600, got {0}'.format(\
                maxgapsec))

    # Determines start and end of hourly series
    start = se.index[0]
    hstart = datetime(start.year, start.month, start.day, \
                        start.hour)+delta(hours=1)
    ref = datetime(1970, 1, 1)
    hstartsec = np.int32((hstart-ref).total_seconds())

    end = se.index[-1]
    hend = datetime(end.year, end.month, end.day, \
                        end.hour)-delta(hours=1)
    nvalh = np.int32((end-start).total_seconds()/3600)
    hvalues = np.nan*np.ones(nvalh, dtype=np.float64)

    # Run C function
    ierr = c_hydrodiy_data.var2h(maxgapsec, hstartsec, display, \
                varsec, varvalues, hvalues)

    if ierr > 0:
        raise ValueError('c_hydrodiy_data.var2h returns {0}'.format(ierr))

    # Convert to hourly series
    dt = pd.date_range(hstart, freq='H', periods=nvalh)
    hvalues = pd.Series(hvalues, index=dt)

    return hvalues


def hourly2daily(se, start_hour=9, timestamp_end=True, how='mean'):
    ''' Convert an hourly time series to daily

    Parameters
    -----------
    se : pandas.Series
        Hourly time series
    start_hour : int
        Hour of the day when the aggregation starts
    timestamp_end : bool
        Affect the aggregated value at the end of the
        timestep
    how : str
        Aggregation method: mean/sum

    Returns
    -----------
    sed : pandas.Series
        Daily series
   '''
    if not how in ['mean', 'sum']:
        raise ValueError('Expected "how" in [mean/sum], got {0}'.format(\
            how))

    # lag the hourly time series depending
    lag = 24-start_hour if timestamp_end else -start_hour
    sel = se.shift(lag)

    # Aggregate to daily
    aggfun = np.mean if how == 'mean' else np.sum
    if PYVERSION == 2:
        sed = sel.resample('D', how)
    elif PYVERSION == 3:
        sed = sel.resample('D').apply(lambda x: aggfun(x.values))

    return sed


def ratfunc_approx(x, xi, fi, eps=1e-8):
    ''' Approximation by rational function using a barycentric
    expression following

    J.-P. Berrut, Rational functions for guaranteed
    and experimentally wellconditioned
    global interpolation, Comput. Math. Appl. 15 (1988), no. 1, 1-16.

    Parameters
    -----------
    x : numpy.ndarray
        Interpolation points
    xi : numpy.ndarray
        Interpolant abscissae
    fi : numpy.ndarray
        Interpolant values
    eps : float
        Small shift added to the difference between x and xi
        to avoid numerical error when inversing the difference.

    Returns
    -----------
    y : numpy.ndarray
        Interpolated values
   '''

    # Check inputs
    xi = np.atleast_1d(xi)
    fi = np.atleast_1d(fi)
    x = np.atleast_1d(x)

    nxi = len(xi)
    if len(fi) != nxi:
        raise ValueError('Expected fi of length {0}, got {1}'.format(\
            nxi, len(fi)))

    # Difference between x and xi
    # including the +1/-1 weighting
    w = np.ones(nxi)
    w[::2] = -1
    d = x[:, None]-xi[None, :]+eps
    d *= w[None, :]

    # function values
    f = np.repeat(fi[None, :], len(x), 0)

    # Results
    num = np.sum(f/d, axis=1)
    denom = np.sum(1/d, axis=1)

    return num/denom



