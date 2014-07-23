import matplotlib
matplotlib.use('Agg')

import datetime 
from dateutil.relativedelta import relativedelta as reldel
import os
import json

import numpy as np
from matplotlib import pyplot as plt
from wafari import view as w
import pandas as pd

#--------------------------------------------------  
# funcs
#--------------------------------------------------  
def water_year(dt, start_month = 7):
    m = dt.month
    if m >= 1 and m < start_month:
        return dt.year-1
    else:
        return dt.year


def concat(series):
    idx = list(series[0].index)
    for i in range(1,len(series)):
        idx = idx + list(series[i].index)
    idx = np.sort(np.unique(idx))

    data = pd.DataFrame(index=idx)
    if type(idx[0]) is datetime.date:
         data['month'] = [x.month for x in idx]
         data['year'] = [x.year for x in idx]
         data['water_year'] = [water_year(x) for x in idx]

    for i in range(len(series)):
        data[series[i].name] = series[i]

    return data


#--------------------------------------------------  
# initialise
#--------------------------------------------------  
PROJECT = '/ehpdata/jlerat/project_jul' 
FREQUENCY = 'monthly'

w.sys.project(PROJECT)
w.sys.frequency(FREQUENCY)
config = json.load(file('%s/wafari/report_dm.json'%PROJECT))
basin = 'murrumbidgee'
catchment = 'tinderry'
id = '410734'

chIMG = '%s/images/%s'%(PROJECT, id)
if not os.path.exists(chIMG): 
    os.makedirs(chIMG)

#--------------------------------------------------  
# check data
#--------------------------------------------------  
w.sf.ingest(ID=id, frequency=FREQUENCY)
q = w.sf.grab(ID=id, frequency=FREQUENCY)
flow = pd.Series(q[1], index=q[0], name='runoff_mmpM')

w.hm.ingest(ID=id, frequency=FREQUENCY)
p = w.hm.grab(ID=id, frequency=FREQUENCY, variable_type='PRECIPITATION')
rainfall = pd.Series(p[1], index=p[0], name='rainfall_mmpM')

pe = w.hm.grab(ID=id, frequency=FREQUENCY, variable_type='POTENTIAL_EVAPORATION')
evap = pd.Series(pe[1], index=pe[0], name='evap_mmpM')

data = concat([flow, rainfall, evap])
start = min(data.index[~np.isnan(data['runoff_mmpM'])]) - reldel(days=365*5)
data = data[data.index>=start]

m = data.groupby('month').mean()
y = data.groupby('water_year').sum()
y['aridity'] = y['rainfall_mmpM']/y['evap_mmpM']
y['runoff_coeff'] = y['runoff_mmpM']/y['rainfall_mmpM']

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

ax1.plot(y['aridity'], y['runoff_coeff'], 'o')
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
xx = np.linspace(0, 4, 1000)
yy = 1-1/xx
ax1.plot(xx, yy, ':')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_title('Adimensional water balance plot')
ax1.set_xlabel('Aridity index P/PE (-)')
ax1.set_ylabel('Runoff coefficient Q/P (-)')


ax2.plot(xx, yy, ':')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_title('Adimensional water balance plot')
ax2.set_xlabel('Aridity index P/PE (-)')
ax2.set_ylabel('Runoff coefficient Q/P (-)')




plt.show()
#plt.savefig('%s/aridity_vs_runoffcoeff.png'%chIMG)

#--------------------------------------------------  
# calibrate
#--------------------------------------------------  
w.sys.basin(basin)
w.sys.catchment(catchment)



