import datetime
import numpy as np
import pandas as pd

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

    return clim


