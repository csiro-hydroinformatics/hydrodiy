import re
import os
import json
from datetime import datetime, date
import numpy as np
import pandas as pd

from hyio import csv

import tempfile

FTMP = tempfile.gettempdir()

URL_EHPDB = 'http://wafaridev/data/jenkins/ehpdb-vis-web/content'


def __download(f1, f2):
    os.system('wget %s -O %s' % (f1, f2))


def get_sites():

    f1 = '%s/config/site_config.json' % URL_EHPDB
    f2 = '%s/sites.json' % FTMP
    __download(f1, f2)
    js = json.load(open(f2, 'r'))
    os.remove(f2)
    
    si = [s['properties'] for s in js['stations']['features']]
    sites = {}
    for s in si:
        sites[s['AWRC_ID']] = s

    sites = pd.DataFrame(sites).T

    return sites

    

def get_monthlyflow(id):

    qobs = []

    for month in range(1, 13):
        f1 = '%s/data/%s/%s_monthly_total_%2.2d.csv' % (URL_EHPDB, id, id, month)
        f2 = '%s/monthly.csv' % FTMP
        __download(f1, f2)
   
        if os.path.exists(f2):
            d, comment = csv.read_csv(f2)
            os.remove(f2)

            # process
            d['month'] = pd.to_datetime(['%d-%2.2d-01' % (y, month) 
                        for y in d['Year']])
            d = d.set_index('month')
            qobs.append(d.iloc[:,1])

    if len(qobs)>1:
        qobs = pd.concat(qobs)
        qobs = qobs.sort_index()

    return qobs


def get_daily(id):

    daily = []

    f1 = '%s/data/%s/%s_daily_ts.csv' % (URL_EHPDB, id, id)
    f2 = '%s/daily.csv' % FTMP
    __download(f1, f2)
   
    if os.path.exists(f2):
        daily, comment = csv.read_csv(f2, index_col=0, parse_dates=True)
        os.remove(f2)

    return daily
