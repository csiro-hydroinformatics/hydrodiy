""" Utility functions to process data """
import re
from datetime import datetime
from calendar import month_abbr
from dateutil.relativedelta import relativedelta as delta

import numpy as np
import pandas as pd

from pandas.tseries.offsets import DateOffset
from numpy.polynomial import polynomial as poly

from hydrodiy import has_c_module
if has_c_module("data", False):
    import c_hydrodiy_data

def sub(text, to_replace):
    """ Replace in "text" all occurences of any key in the
    dictionary "to_replace" by its corresponding value.

    Parameters
    -----------
    text : str
        Input string.
    to_replace : dict
        Pattern to match and replace.
        Use an ordered dict (collections.OrderedDict) to
        replace in a certain order.

    Returns
    -----------
    replaced : str
        Processed string.
    """
    replaced = text
    for pattern, rep in to_replace.items():
        replaced = re.sub(pattern, rep, replaced)

    return replaced


def sequence_true(values):
    """ Identify start and end of consecutive "true" values.
    Can be used for gap analysis.

    Parameters
    -----------
    values : numpy.ndarray
        Vector of booleans

    Returns
    -----------
    startend : numpy.ndarray
        Indexes of sequence starts (column 1) and ends (column 2)
    """
    values = values.astype(int)
    values_filled = np.append(0, np.append(values, 0))
    diff = np.diff(values_filled)

    start = np.where(diff == 1)[0]
    end = np.where(diff == -1)[0]
    startend = np.column_stack([start, end])

    return startend


def cast(x, y):
    """ Cast y to the type of x.

        Useful to make sure that a function returns an output that has
        the same type than the input (e.g. to avoid mixing float with
        numpy.array 0d).

    Parameters
    -----------
    x : object
        First object
    y : object
        Second object
    """
    # Check if x is an nxd numpy array (n>0)
    # then collect is type
    xdtype = None
    if hasattr(x, "dtype"):
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
            raise TypeError("Cannot cast value from float to int")

        ycast = type(x)(y)

    else:
        # x is a numpy array
        ycast = np.array(y).astype(xdtype, casting="safe")

    return ycast


def dayofyear(days):
    """ Compute day of year using days.dayofyear, but reduce the value by
    one for leap-years. This ensures that all years have 365 days.

     Parameters
    -----------
    days : pandas.core.indexes.dateteime.DatetimeIndex
        Date series

    Returns
    -----------
    doy : numpy.ndarray
        Day of year
    """
    try:
        doy = days.dayofyear.values
        yy = days.year.values
        mm = days.month.values
    except AttributeError:
        # Allow older version of pandas to work
        doy = days.dayofyear
        yy = days.year
        mm = days.month

    isleap = (yy % 4 == 0) & (~(yy % 100 == 0) | (yy % 400 == 0))
    idx = (mm > 2) & isleap
    doy[idx] = doy[idx]-1

    return doy


def compute_aggindex(time, timestep):
    """ Compute the aggregation index for certain aggregation times.

    Parameters
    -----------
    time : pandas.DatetimeIndex
        Data time stamps.
    """
    if timestep.startswith("AS"):
        if timestep != "AS":
            mth = re.sub("AS-", "", timestep)
            allowed = [month_abbr[i].upper() for i in range(1, 13)]
            errmsg = f"Expected month in {'/'.join(allowed)}, got"+\
                        f" timestep={timestep}."
            assert mth in allowed, errmsg
    else:
        allowed = ["D", "H", "MS"]
        errmsg = f"Expected time step in {'/'.join(allowed)}, got {timestep}."
        assert timestep in allowed, errmsg

    if timestep == "AS":
        return time.year.values
    elif timestep.startswith("AS") and timestep != "AS":
        imth = 11-np.where([mth == m for m in allowed])[0][0]
        return (time+pd.DateOffset(months=imth)).year.values-1
    elif timestep == "MS":
        return time.year*100+time.month
    elif timestep == "D":
        return time.year*10000+time.month*100+time.day
    elif timestep == "H":
        return time.year*1000000+time.month*10000\
                    +time.day*100+time.hour


def aggregate(aggindex, inputs, oper=0, maxnan=0):
    """ Fast aggregation of inputs based on aggregation indices
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

    """
    has_c_module("data")

    # Check inputs
    if len(aggindex) != len(inputs):
        raise ValueError("Expected same length for aggindex and"+\
                    f" inputs,got len(aggindex)={len(aggindex)} "+
                    f"and len(inputs)={len(inputs)}.")

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
        raise ValueError("c_hydrodiy_data.aggregate"+\
                            " returns {0}".format(ierr))

    # Truncate the outputs to keep only the valid part of the vector
    outputs = outputs[:iend[0]]

    return outputs


def flathomogen(aggindex, inputs, maxnan=0):
    """ Compute a series of the same length than inputs with all values
    replaced by mean computed for each value of the aggregation index.

    This function is used in hydrodiy.data.signatures.goue

    Parameters
    -----------
    aggindex : numpy.ndarray
        Aggregation index (e.g. month in the form 199501 for
        Jan 1995). Index should be in increasing order!
    inputs : numpy.ndarray
        Inputs data to be aggregated
    maxnan : int
        Maximum number of nan in inputs for each
        aggregation index

    Returns
    -----------
    outputs : numpy.ndarray
        Flat disaggregated data

    """
    has_c_module("data")

    # Check inputs
    if len(aggindex) != len(inputs):
        raise ValueError("Expected inputs of length "+\
                f"{len(aggindex)}, got {len(inputs)}.")

    # Allocate arrays
    maxnan = np.int32(maxnan)
    aggindex = np.array(aggindex).astype(np.int32)
    inputs = inputs.astype(np.float64)
    outputs = 0.*inputs

    # Run C function
    ierr = c_hydrodiy_data.flathomogen(maxnan, aggindex, \
                inputs, outputs)

    if ierr > 0:
        raise ValueError("c_hydrodiy_data.flatdisagg"+\
                            " returns {0}".format(ierr))

    return outputs


def lag(data, lag, missing=np.nan):
    """ Lag a numpy array and adds NaN at the beginning or end of the
        lagged data depending on the lag value. The lag is introduced
        using numpy.roll

        This is equivalent to pandas.DataFrame.shift function, but for Numpy
        arrays.

    Parameters
    -----------
    data : numpy.ndarray data
        Data series where linear interpolation is suspected
    lag : int
        Lag. If >0, lag the data forward. If<0 lag the data backward.
    missing : float
        Missing value pasted at the end or the beginning of the data.

    Returns
    -----------
    lagged : numpy.ndarray
        Lagged vector

    """
    # Check data
    data = np.atleast_1d(data)
    lag = int(lag)

    # Lagg data
    lagged = np.roll(data, lag, axis=0)
    if lag < 0:
        lagged[lag:] = missing
    elif lag > 0:
        lagged[:lag] = missing
    else:
        lagged = data.copy()

    return lagged


def water_year(time, start_month=7):
    """ Define the water year for a set of time stamps.

    Parameters
    -----------
    time : pandas.DatetimeIndex
        Data time stamps.
    start_month : int
        Water year start month.
    """
    errmsg = f"Expected start month in [1, 12], got {start_month}."
    assert start_month>=1 and start_month<=12, errmsg

    years = lag(time.year.values, start_month-1, -1)

    # Fill in missing values
    missing = years < 0
    y1 = years[~missing].min()
    years[missing] = y1-1

    return years


def monthly2daily(se, interpolation="flat", minthreshold=0.):
    """ Convert monthly series to daily using interpolation.
    Takes care of the boundary effects.

    Parameters
    -----------
    sem : pandas.core.series.Series
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
    """
    # Set
    sec = se.copy()
    sec[pd.isnull(sec)] = minthreshold-1

    if interpolation == "flat":
        # Add a fictive data after the last month
        # to allow for resample to work
        nexti = sec.index[-1] + delta(months=1)
        sec[nexti] = np.nan

        # Convert to daily
        sed = sec.resample("D").fillna(method="pad")
        sed /= sed.index.days_in_month
        sed[sed < minthreshold] = np.nan

        # Drop last values
        sed = sed.iloc[:-1]

    elif interpolation == "cubic":
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
        # 3. f"(0) = d0
        # 4. f"(1) = d1
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
        raise ValueError("Expected interpolation in [flat/cubic], got "+\
                    interpolation)

    return sed


def var2h(se, maxgapsec=5*86400, display=False):
    """ Convert a variable time step time series to hourly using
        linear interpolation and aggregation

    Parameters
    -----------
    se : pandas.core.series.Series
        Irregular time series
    maxgapsec : int
        Maximum number of seconds between two valid measurements
    display : bool
        Display progresses or not

    Returns
    -----------
    seh : pandas.Series
        Hourly series
    """
    has_c_module("data")

    # Allocate arrays
    maxgapsec = np.int32(maxgapsec)
    if maxgapsec < 3600:
        raise ValueError(f"Expected maxgapsec>=3600, got {maxgapsec}.")

    display = np.int32(display)
    varvalues = se.values.astype(np.float64)

    time = se.index.tz_localize(None).values
    varsec = np.int64(time.astype(np.int64)/1000000000)

    # Determines start and end of hourly series
    start = se.index[0]
    hstart = datetime(start.year, start.month, start.day, \
                        start.hour)+delta(hours=1)
    ref = datetime(1970, 1, 1)
    hstartsec = np.int64((hstart-ref).total_seconds())

    end = se.index[-1]
    hend = datetime(end.year, end.month, end.day, \
                        end.hour)-delta(hours=1)
    nvalh = np.int32((end-start).total_seconds()/3600)
    hvalues = np.nan*np.ones(nvalh, dtype=np.float64)

    # Run C function
    ierr = c_hydrodiy_data.var2h(maxgapsec, hstartsec, display, \
                varsec, varvalues, hvalues)

    if ierr > 0:
        raise ValueError(f"c_hydrodiy_data.var2h returns {ierr}")

    # Convert to hourly series
    dt = pd.date_range(hstart, freq="H", periods=nvalh)
    hvalues = pd.Series(hvalues, index=dt)

    return hvalues


def oz_timezone(lon, lat):
    """ Returns the time zone in Australia for a particular location.
    Does not take into account the Eastern border NSW/QLD along
    Border Rivers.

    Parameters
    -----------
    lon : float
        Longitude
    lat : float
        Latitude

    Returns
    -----------
    tz : string
        Time zone compatible with pytz package naming convention.
    """
    if lon < 129.:
        return "Australia/Perth"
    elif lon < 141.:
        if lat < -26.:
            return "Australia/Adelaide"
        else:
            if lon > 138.:
                return "Australia/Brisbane"
            else:
                return "Australia/Darwin"
    else:
        if lat < -29:
            return "Australia/Sydney"
        else:
            return "Australia/Brisbane"


