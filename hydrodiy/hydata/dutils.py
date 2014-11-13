from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from hystat import sutils

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
     
def runclim(data, nwin=10):
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
    for d in doy:
        dist = cycledist(doyidx, d, 1, 365)
        idx =  (dist<=nwin) & (~no)
        df = pd.DataFrame(dict(data[idx].describe()), index=[d])

        if clim is None:
            clim = df
        else:
            clim = clim.append(df)

    clim = clim.rename(columns={'50%':'median'})
    clim['day'] = pd.date_range('2001-01-01', freq='D', periods=365)

    # Find beginning of water year
    ts = pd.Series(clim['median'].values,  index=clim['day'])
    climm = ts.resample('MS') 
    rm = relativedelta(months=6)
    dtmax = climm.index[climm==np.max(climm)].values[0]
    dtmax = datetime.strptime(dtmax.astype(str)[:10], '%Y-%m-%d')
    wateryear_startmonth = (dtmax-rm).month

    return clim, wateryear_startmonth

def runclimcum(data, clim, wateryear_startmonth):
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
    data2['year'] -= data.index[0].month<=wateryear_startmonth
    datat = pd.pivot_table(data2, rows='nday', 
            cols='year', values='data')

    # compute cum sum
    datat = datat.cumsum(axis=0)

    # Compute stats
    climc =  datat.apply(lambda x: x.describe(), axis=1)
    climc = climc.rename(columns={'50%':'median'})

    # Correct for decreasing trends in clim
    # remove decreasing streches and rescale to keep overall balance
    def fix(x):
        dd = x.diff().clip(0., 1e30)
        dd[:1] = x[:1]
        xx = dd.cumsum()
        return xx/xx.sum()*x.sum()
    climc = climc.apply(fix)

    # Find lowest/highest year
    total = datat.loc[datat.index[-1],:]
    ymin = datat.columns[np.where(total==np.min(total))[0][0]]
    climc['lowest'] = datat.loc[:, ymin]
    ymax = datat.columns[np.where(total==np.max(total))[0][0]]
    climc['highest'] = datat.loc[:, ymax]

    climc = climc.set_index(clim.index)

    return climc, datat

def to_seasonal(ts):
    ''' Convert time series to seasonal time step '''
    
    # Resample to monthly
    tsm = ts.resample('MS', 'sum')

    # Produce 3 shifted series
    tss = []
    for s in range(0, 3):
        tss.append(tsm.shift(-s))

    tss = pd.DataFrame(tss)
    tss.loc[:,tss.columns[-2:]] = np.nan

    return tss.apply(np.sum, axis=0)

