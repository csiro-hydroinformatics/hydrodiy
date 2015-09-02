import re
import math

from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

import numpy as np
import pandas as pd

from hystat import sutils

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

def time2osec(t):
    ''' Convert date/time object to ordinal seconds 

    Parameters
    -----------
    t : datetime.datetime
        Time to be converted

    Returns
    -----------
    o : numpy.uint64
        Number of seconds elapsed since 0000-00-00 00:00:00
        
    Example
    -----------
    >>> import datetime
    >>> dutils.time2osec(datetime.datetime(2000,10, 1, 10))
    63106077600

    '''

    o = np.uint64(t.toordinal() * 86400)
    o += np.uint64(t.hour * 3600)
    o += np.uint64(t.minute * 60)
    o += np.uint64(t.second)

    return o

def secofyear(t):
    ''' 
        Compute number of seconds elapsed since Jan 1

    Parameters
    -----------
    t : datetime.datetime
        Time to be converted

    Returns
    -----------
    n : numpy.uint64
        Number of seconds elapsed since the first of January of the same year
        
    Example
    -----------
    >>> import datetime
    >>> dutils.secofyear(datetime.datetime(2000,10, 1, 10))
    23623200

    '''

    o = time2osec(t)
    o0 = time2osec(datetime(t.year, 1, 1))

    if calendar.isleap(t.year) & (t.month >= 3):
        o -= np.uint64(86400)

    n = o-o0

    return n


def osec2time(o):
    ''' 
        Convert ordinal seconds to date/time 

    Parameters
    -----------
    o : numpy.uint64
        Ordinal second (number of sec elapsed from 0000/00/00 00:00:00)

    Returns
    -----------
    t : datetime.datetime
        Time corresponding to o
        
    Example
    -----------
    >>> import datetime
    >>> t = datetime.datetime(2000,10, 1, 10)
    >>> o = dutils.time2osec(t)
    >>> t2 = dutils.osec2time(o)
    >>> t == t2
    True

    '''
   
    oi = int(o)/86400
    d = datetime.fromordinal(oi)
    
    hour = int(o - oi*86400)/3600
    minute = int(o - oi*86400 - hour*3600)/60
    second = int(o - oi*86400 - hour*3600 - minute*60)

    d = datetime(d.year, d.month, d.day, hour, minute, second)

    return d


def wyear(dt, start_month=7):
    ''' compute water year of a particular day '''
    return dt.year - int(dt.month<start_month)


def wyear_days(dt, start_month=7):
    ''' compute number of days from start of the water year '''
    yws = datetime(wyear(dt, start_month), start_month, 1)
    return (dt-yws).days+1


def cycledist(x, y, start=1, end=12):
    ''' Compute abs(x-y) assuming that x and y are following a cycle (e.g. 12 months with start=1 and end=12)'''
    cycle = float(end-start+1)
    return np.abs((x-y-cycle/2)%cycle-cycle/2)


def padclim(clim, nwin, is_cumulative=False):
    ''' Pad start and end climate dataset '''

    # Pad data at start
    start = clim.tail(nwin*2).copy()
    dstart = clim.index[0].to_datetime() - relativedelta(days=nwin*2)
    days = pd.date_range(dstart, periods=nwin*2, freq='D')
    start = start.set_index(days)

    # Set negative values if data is cumulative
    if is_cumulative:
        start = start.apply(lambda x: x-x[-1])

    # Pad data at the end
    end = clim.head(nwin*2).copy()
    dend = clim.index[-1].to_datetime() + relativedelta(days=1)
    days = pd.date_range(dend, periods=nwin*2, freq='D')
    end = end.set_index(days)
    
    # Set negative values if data is cumulative
    if is_cumulative:
        end = end + np.dot(np.ones((end.shape[0], 1)), clim[-1:].values) 

    clim = clim.append([start, end])
    clim = clim.sort()

    return clim

def runclim(data, nwin=20, ispos=True, perc=[5, 10, 25, 50, 75, 90, 95]):
    ''' Compute climatology of a daily time series 
        
    '''
    doy = np.arange(1, 366)
    clim = None

    # Deals with leap years
    dx = data.index
    no = (dx.month==2) & (dx.day==29) 
    doyidx = dx.dayofyear
    correc = (dx.month>2) & (dx.year%4==0)
    doyidx[correc] = doyidx[correc] - 1

    # Compute running stat
    clim = {}
    for d in doy:
        dist = cycledist(doyidx, d, 1, 365)
        idx =  (dist<=nwin) & (~no) 
        if ispos: idx = idx & (data>=0)
        clim[d] = sutils.percentiles(data[idx], perc)

    clim = pd.DataFrame(clim).T
    clim.columns = ['%d%%'%p for p in perc]
    days = pd.date_range('2001-01-01', freq='D', periods=365)
    clim = clim.set_index(days)

    # pad start and end
    clim = padclim(clim, nwin)

    # Apply rectangular filter and clip to proper dates
    clim = pd.rolling_mean(clim, window=nwin)
    clim = clim.shift(-nwin/2)
    clim = clim[clim.index.year==2001]

    # Find beginning of water year
    climm = clim['50%'].resample('MS') 
    dtmax = climm.index[climm==np.max(climm)].values[0]
    dtmax = datetime.strptime(dtmax.astype(str)[:10], '%Y-%m-%d')
    rm = relativedelta(months=6)
    wateryear_startmonth = (dtmax-rm).month

    return clim, wateryear_startmonth


def runclimcum(data, clim, wateryear_startmonth, nwin=20):
    ''' Compute cumulative climatology '''

    # Gapfill timeseries with monthly median
    def replace(grp):
        idx = pd.isnull(grp)
        grp[idx] = grp[~idx].median()
        return grp

    data = data.groupby(data.index.month).transform(replace)

    # reorganise series to exclude leap years
    nope = (data.index.month==2) & (data.index.day==29)
    dx = data[~nope].index
    dd = pd.DataFrame({'1':dx.month, '2':dx.day})
    dts = datetime(2001, wateryear_startmonth, 1)

    def fun(x):
        if x[0]>=wateryear_startmonth:
            return (datetime(2001, x[0], x[1])-dts).days
        else:
            return (datetime(2002, x[0], x[1])-dts).days

    data2 = pd.DataFrame({'data':data[~nope].values, 
            'nday':dd.apply(fun, axis=1).values},
            index=data.index[~nope])

    data2['year'] = (data2['nday']==1).astype(int)
    data2['year'] = data2['year'].cumsum() 
    data2['year'] += data.index[0].year
    data2['year'] -= data.index[0].month <= wateryear_startmonth
    datat = pd.pivot_table(data2, index='nday', 
            columns='year', values='data')

    # compute cum sum
    datat = datat.cumsum(axis=0)

    # Compute stats
    perc = [int(re.sub('%', '', cn)) for cn in clim.columns if cn.endswith('%')]
    climc =  datat.apply(lambda x: sutils.percentiles(x, perc), axis=1)
    climc.columns = ['%d%%'%p for p in perc]

    idx = pd.date_range('%4d-%2.2d-01' % (2001, wateryear_startmonth), 
                freq='D', periods=365)

    climc = climc.set_index(idx)

    # Correct for decreasing trends in clim
    # remove decreasing streches and rescale to keep overall balance
    def fix(x):
        dd = x.diff().clip(0., np.inf)
        dd[:1] = x[:1]
        xx = dd.cumsum()
        return xx/xx.sum()*x.sum()

    climc = climc.apply(fix)

    # Pad start/end
    climc = padclim(climc, nwin, True)

    # Apply rolling mean to smooth out data
    climc = pd.rolling_mean(climc, window=nwin)
    climc = climc.shift(-nwin/2)
    climc = climc[(climc.index>=idx[0]) & (climc.index<=idx[-1])]
    climc = climc.apply(lambda x: np.clip(x, 0., np.inf))

    # Find lowest/highest year
    total = datat.loc[datat.index[-1],:]
    ymin = datat.columns[np.where(total == np.min(total))[0][0]]
    climc['lowest'] = datat.loc[:, ymin]
    ymax = datat.columns[np.where(total == np.max(total))[0][0]]
    climc['highest'] = datat.loc[:, ymax]

    return climc, datat

def to_seasonal(ts, nmonths=3):
    ''' Convert time series to seasonal time step

    Parameters
    -----------
    ts : pandas.core.series.Series
        Input time series
    nmonths : int
        Number of months used for aggregation

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
    tsm = ts.resample('MS', 'sum')

    # Shift series
    tss = []
    for s in range(0, nmonths):
        tss.append(tsm.shift(-s))

    tss = pd.DataFrame(tss)
    out = tss.sum(axis=0, skipna=False)

    return out

def tofloat(x):
    try:
        f = float('%s' % x)
    except ValueError:
        f = np.nan

    return f

def tofloats(s):
    return pd.Series([tofloat(x) for x in s], index=s.index)


def atmospress(altitude):
    ''' Mean atmospheric pressure 
        See http://en.wikipedia.org/wiki/Atmospheric_pressure
    '''

    g = 9.80665 # m/s^2 - Gravity acceleration
    M = 0.0289644 # kg/mol - Molar mass of dry air
    R = 8.31447 # j/mol/K - Universal gas constant
    T0 = 288.15 # K - Sea level standard temp
    P0 = 101325 # Pa - Sea level standard atmospheric pressure

    P = P0 * math.exp(-g*M/R/T0 * altitude) 

    return P
