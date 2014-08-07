import datetime
import numpy as np
import pandas as pd

from hystat import sutils

def wyear(dt, start_month=7):
    ''' compute water year of a particular day '''
    return dt.year - int(dt.month<start_month)

def wyear_days(dt, start_month=7):
    ''' compute number of days from start of the water year '''
    yws = datetime.datetime(wyear(dt, start_month), start_month, 1)
    return (dt-yws).days+1

def cycledist(x, y, start=1, end=12):
    ''' Compute abs(x-y) assuming that x and y are following a cycle (e.g. 12 months with start=1 and end=12)'''
    cycle = float(end-start+1)
    return np.abs((x-y-cycle/2)%cycle-cycle/2)

#def get_seasonal(data):
#    ''' Return 3 monthly aggregated time series 
#        
#        :param pandas.Series data: Input data (assuming daily)
#    '''
#    o = d.resample('MS', how='sum')
#    o['lag'] = o['value'].shift(1)
#    
#    if timestep=='seasonal':
#        o3a = o.resample('3MS', how='sum')
#        o3a['lag'] = o3a['value'].shift(1)
#        o3b = o[1:].resample('3MS', how='sum')
#        o3b['lag'] = o3b['value'].shift(1)
#        o3c = o[2:].resample('3MS', how='sum')
#        o3c['lag'] = o3c['value'].shift(1)
#        o = pd.concat([o3a, o3b, o3c])
#        o = o.sort()
     
def runclim(data, nwin=10):
    ''' Compute climatology of a daily time series 
        
    '''
    doy = np.arange(1, 367)
    clim = None

    for d in doy:
        dist = cycledist(data.index.dayofyear, d, 1, 366)
        idx =  dist<=nwin
        df = pd.DataFrame(dict(data[idx].describe()), index=[d])

        if clim is None:
            clim = df
        else:
            clim = clim.append(df)

    clim = clim.rename(columns={'50%':'median'})

    return clim

def runclimcum(data, clim=None, nwin=10):
    ''' Gapfill time series '''

    # Get climatology
    if clim is None:
        clim = runclim(data, nwin)

    # Find beginning of water year
    ts = pd.Series(clim['median'].values, 
            index = pd.date_range('2000-01-01', freq='D', periods=366))
    climm = ts.resample('MS') 
    start = (climm.index[climm==np.max(climm)].dayofyear-183)%366

    # reorganise series
    ts = pd.DataFrame({'data':data.values, 
            'dayfromstart':1+(data.index.dayofyear-start)%366})    

    ts['year'] = ts['dayfromstart']==1
    ts['year'] = ts['year'].astype('int').cumsum()
    tsc = pd.pivot_table(ts, rows='dayfromstart', 
            cols='year', values='data')

    # Gapfill
    for i in tsc.index:
        val = clim['median'][(i-1+start)%366].values[0]
        ccx = pd.isnull(tsc.loc[i,:])
        tsc.loc[i,ccx] = val

    # compute cum sum
    tsc = tsc.cumsum(axis=0)

    # Compute stats
    out =  tsc.apply(lambda x: x.describe(), axis=1)
    out = out.rename(columns={'50%':'median'})

    # Find lowest/highest year
    total = tsc.loc[366,:]
    out['lowest'] = tsc.loc[:, total==np.min(total)]
    out['highest'] = tsc.loc[:, total==np.max(total)]

    return out


