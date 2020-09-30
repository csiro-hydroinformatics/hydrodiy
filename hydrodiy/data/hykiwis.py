import os, re
import json
import requests
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

from hydrodiy import PYVERSION

from hydrodiy.data.qualitycontrol import islinear
from hydrodiy.io import csv

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                        'data')

#
# This code is largely pasted from the kiwis_py package
# developed by Andrew McDonald, Bureau of Meteorology, EHP
#

KIWIS_URL_EXT = 'http://www.bom.gov.au/waterdata/KiWIS/KiWIS'
KIWIS_URL_INT = 'http://wiski-prod-kiwis01:8080/KiWIS/KiWIS'

# Base parameters for Kiwis server request
BASE_PARAMS = {\
    'service':'kisters', \
    'type':'queryServices', \
    'datasource':0, \
    'format':'json' \
}

# Kiwis time series names
TS_NAMES = {\
    'as_stored': 'DMQaQc.Merged.AsStored.1', \
    'hourly': 'DMQaQc.Merged.HourlyMean.HR', \
    'daily_0am': 'DMQaQc.Merged.DailyMean.24HR', \
    'daily_9am': 'DMQaQc.Merged.DailyMean.09HR', \
    'daily_9am_qa': 'PR01QaQc.Merged.DailyMean.09HR', \
    'daily_min': 'DMQaQc.Merged.DailyMin.24HR', \
    'daily_max': 'DMQaQc.Merged.DailyMax.24HR', \
    'monthly': 'DMQaQc.Merged.MonthlyMean.CalMonth', \
    'yearly': 'DMQaQc.Merged.YearlyMean.CalYear'
}

# Default start year for data download
START_YEAR = 1950


def __testjson(req):
    ''' Test validity of json conversion '''
    try:
        out = req.json()
        return out

    except ValueError as jerr:
        warnings.warn('Repairing json text')
        txt = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', \
                                r'', req.text)

        if re.search('Unavailable', txt):
            return {'error': 'Service unavailable'}
        else:
            return json.loads(txt)

    except Exception as err:
        warnings.warn('Formatting of json data return {0}'.format(err))
        return None


def has_internal_access():
    ''' Test if internal access is possible '''

    url = re.sub('KiWIS$', '', KIWIS_URL_INT)
    try:
        req = requests.get(url)
        req.raise_for_status()
        return True
    except Exception as err:
        return False


def get_storages():
    ''' Get list of storages in Australia

    Returns
    -----------
    storages : pandas.core.frame.DataFrame
        List of storages
    '''
    fs = os.path.join(DATA_FOLDER, 'storages.csv')
    storages, _ = csv.read_csv(fs, index_col='kiwisid')
    return storages


def get_sites(download=False):
    ''' Get list of BoM Kiwis stations

    Parameters
    -----------
    download : bool
        If True, download data from Bureau of Meterology internal Kiwis server
        and store it locally (accessible within Bureau network only).
        If False, just reads local data.

    Returns
    -----------
    sites : pandas.core.frame.DataFrame
        List of BoM Kiwis sites
    url : str
        Url used to query the Kiwis server
    '''
    # path to csv file
    fsites = os.path.join(DATA_FOLDER, 'kiwis_sites.csv')

    # Download site list
    fsites_zip = re.sub('csv$', 'zip', fsites)
    if download or not os.path.exists(fsites_zip):
        params = dict(BASE_PARAMS)
        params.update({ \
                    'request': 'getStationList', \
                    'returnfields':'station_no,station_name,'+\
                        'station_longname,station_id,' +\
                        'object_type,station_latitude,station_longitude' \
        })

        url = KIWIS_URL_INT
        req = requests.get(url, params=params)

        # Format list of sites
        sites = __testjson(req)
        if sites is None or re.search('No matches', ' '.join(sites[0])):
            raise ValueError('Request returns no data. URL={0}'.format(req.url))

        sites = pd.DataFrame(sites[1:], columns = sites[0])

        comments = {'data': 'Kiwis sites', \
                    'date_downloaded': str(datetime.now().date())}
        csv.write_csv(sites, fsites, comments, \
                os.path.abspath(__file__), write_index=True)

        return sites, req.url

    else:
        sites, comments = csv.read_csv(fsites)
        comment = 'No download. list of sites read from local '+\
                    'data downloaded  on {}'.format(comments['date_downloaded'])
        warnings.warn(comment)

        return sites, comment


def get_tsattrs(siteid, ts_name, external=True):
    ''' Retrieve time series meta data from site ID

    Parameters
    -----------
    siteid : str
        Site ID in Kiwis server
    ts_name : str
        Name of the Kiwis time series. Options are:
        * as_stored : Series as provided by the data provider (e.g. streamflow
                                data are continuous)
        * hourly : Hourly aggregated series
        * daily_9am : Daily aggregated series from 9.00 to 9.00.
                        Timestamping at the end of the time step.
        * daily_9am_qa : Same than daily_9am with quality control
        * daily_min : Minimum of provided value over 24 hour periods
        * daily_max : Maximum of provided value over 24 hour periods
        * monthly : Monthly aggregated values
        * yearly : Yearly aggregated values
    external : bool
        Use Bureau of Meterology external Kiwis server.
        If False, use Bureau internal server
        (accessible within Bureau network only).

    Returns
    -----------
    tsattrs_list : list
        List of Kiwis attributes
    url : str
        Url used to query the Kiwis server
    '''

    # Check inputs
    if not ts_name in TS_NAMES:
        raise ValueError('Expected ts_name in {0}, got {1}'.format(\
            list(TS_NAMES.keys()), ts_name))

    if not external:
        if not has_internal_access():
            warnings.warn('Cannot get internal access. '+\
                'Switching to external')
            external = False

    # Download attributes
    params = dict(BASE_PARAMS)
    params.update({
            'request': 'getTimeseriesList',
            'ts_name': TS_NAMES[ts_name],
            'station_no':siteid,
            'returnfields':'ts_id,ts_unitname,ts_unitsymbol,'
                'station_no,station_name,coverage,parametertype_name'
        })

    url = KIWIS_URL_EXT if external else KIWIS_URL_INT
    req = requests.get(url, params=params)

    js_data = __testjson(req)

    # Check outputs
    mess = 'Request returns no data. URL={}'.format(req.url)
    if js_data is None:
        raise ValueError(mess)

    if 'error' in js_data:
        mess = 'Request returns no data. URL={}, Error={}'.format(req.url, \
                                    js_data['error'])
        raise ValueError(mess)

    if 'type' in js_data:
        if js_data['type'] == 'error':
            raise ValueError(mess)

    if re.search('No matches', ''.join(js_data[0])):
        mess = 'Request returns no data, no matches in'+\
                    ' Kiwis server. URL={}'.format(req.url)
        raise ValueError(mess)

    # Format attributes
    tsattrs_list = [{k:v for k, v in zip(js_data[0], js_data[k+1])} \
                    for k in range(len(js_data)-1)]
    for att in tsattrs_list:
        att['ts_name'] = ts_name

    return tsattrs_list, req.url


def get_data(tsattrs, start=None, end=None, external=True, \
                                timezone=None):
    ''' Download time series  data from meta data info

    Parameters
    -----------
    tsattrs : dict
        Kiwis attributes of the timeseries. See hydrodiy.hykiwis.get_tsattrs
    start : str
        Start date of the requested time series including time zone. If None,
        uses the 'from' field from tsattrs.
    end : str
        End date of the requested time series including time zone. If None,
        uses the 'to' field from tsattrs.
    external : bool
        Use Bureau of Meterology external Kiwis server.
        If False, use Bureau internal server
        (accessible within Bureau network only).
    keep_timezone : bool
        Keep time zone information in date parsing

    Returns
    -----------
    ts : pandas.core.series.Series
        Time series
    url : str
        Url used to query the Kiwis server
    '''

    # Check inputs
    if not external:
        if not has_internal_access():
            warnings.warn('Cannot get internal access. '+\
                'Switching to external')
            external = False

    # Download data
    if start is None:
        start = '{0}-01-01'.format(START_YEAR) + tsattrs['from'][10:]

    if end is None:
        end = tsattrs['to']

    params = {'request':'getTimeseriesValues',
        'ts_id':tsattrs['ts_id'],
        'from': start,
        'to': end
    }
    params.update(BASE_PARAMS)
    params['format'] = 'dajson'

    url = KIWIS_URL_EXT if external else KIWIS_URL_INT
    req = requests.get(url, params=params)

    # Convert to pandas series
    js = __testjson(req)

    if js is None:
        raise ValueError('Request returns no data. URL={0}'.format(req.url))

    elif re.search('error', str(js)):
        raise ValueError('Request returns error: {1}. URL={0}'.format(\
                req.url, str(js)))

    else:
        d = pd.DataFrame(js[0]['data'], columns=['time', 'value'])

        # Discard time zone
        if not timezone is None:
            # to_datetime seems to be converting to UTC
            time = pd.to_datetime(d['time'])
            try:
                time = time.dt.tz_localize('UTC')
            except TypeError:
                pass

            # Localise time zone
            time = time.dt.tz_convert(timezone)
        else:
            # Remove all timezone reference
            time = pd.to_datetime(d['time'].apply(lambda x: \
                                    re.sub('\+.*', '', x)))

        ts = d.set_index(time)['value']
        ts.name = '%s[%s]' % (tsattrs['ts_id'], tsattrs['ts_unitsymbol'])

    return ts, req.url

