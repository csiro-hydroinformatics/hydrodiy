import re
import json
import requests
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

from hydrodiy.data.qualitycontrol import islinear

#
# This code is largely pasted from the kiwis_py package
# developed by Andrew McDonald, Bureau of Meteorology, EHP
#

KIWIS_URL_EXT = 'http://www.bom.gov.au/waterdata/KiWIS/KiWIS'
KIWIS_URL_INT ='http://ccfvp-wadiapp04:8080/KiWIS/KiWIS'
KIWIS_URL_SITES = 'http://wiski-04:8080/KiWIS/KiWIS'

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
    'daily_9am': 'DMQaQc.Merged.DailyMean.09HR', \
    'daily_9am_qa': 'PR01QaQc.Merged.DailyMean.09HR'
}

# Default start year for data download
START_YEAR = 1950


def __testjson(req):
    ''' Test validity of json conversion '''

    try:
        out = req.json()
        return out

    except json.decoder.JSONDecodeError as jerr:
        warnings.warn('Repairing json text')
        txt = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', req.text)
        return json.loads(txt)

    except Exception as err:
        warnings.warn('Formatting of json data return {0}'.format(err))
        return None


def get_sites(external=True):
    ''' Get list of BoM Kiwis stations

    Parameters
    -----------
    external : bool
        Use Bureau of Meterology external Kiwis server. If False, use Bureau internal
        server (accessible within Bureau network only).

    Returns
    -----------
    sites : pandas.core.frame.DataFrame
        List of BoM Kiwis sites
    url : str
        Url used to query the Kiwis server
    '''

    # Download site list
    params = dict(BASE_PARAMS)
    params.update({ \
                'request': 'getStationList', \
                'returnfields':'station_no,station_name,station_longname,station_id,' +\
                    'object_type,station_latitude,station_longitude' \
    })

    # TODO: check this url works for external
    url = KIWIS_URL_SITES
    req = requests.get(url, params=params)

    # Format list of sites
    sites = __testjson(req)
    if sites is None or re.search('No matches', ' '.join(sites[0])):
        raise ValueError('Request returns no data. URL={0}'.format(req.url))

    sites = pd.DataFrame(sites[1:], columns = sites[0])

    return sites, req.url


def get_tsattrs(siteid, ts_name='daily_9am_qa', external=True):
    ''' Retrieve time series meta data from site ID

    Parameters
    -----------
    siteid : str
        Site ID in Kiwis server
    ts_name : str
        Name of the Kiwis time series. Options are:
        * daily_9am : Daily aggregated series from 9.00 to 9.00.
                        Timestamping at the end of the time step.
        * daily_9am_qa : Same than daily_9am with quality control
        * as_stored : Series as provided by the data provider (e.g. streamflow
                                data are continuous)
    external : bool
        Use Bureau of Meterology external Kiwis server. If False, use Bureau internal
        server (accessible within Bureau network only).

    Returns
    -----------
    tsattrs : dict
        Kiwis attributes
    url : str
        Url used to query the Kiwis server
    '''

    # Check inputs
    if not ts_name in TS_NAMES:
        raise ValueError('Expected ts_name in {0}, got {1}'.format(\
            list(TS_NAMES.keys()), ts_name))

    # Download attributes
    params = dict(BASE_PARAMS)
    params.update({
            'request': 'getTimeseriesList',
            'ts_name': TS_NAMES[ts_name],
            'station_no':siteid,
            'returnfields':'ts_id,ts_unitname,ts_unitsymbol,'
                'station_no,station_name,coverage'
        })

    url = KIWIS_URL_EXT if external else KIWIS_URL_INT
    req = requests.get(url, params=params)

    tsattrs = __testjson(req)
    if tsattrs is None or re.search('No matches', ''.join(tsattrs[0])):
        raise ValueError('Request returns no data. URL={0}'.format(req.url))

    # Format attributes
    tsattrs = {k:v for k, v in zip(tsattrs[0], tsattrs[1])}
    tsattrs['ts_name'] = ts_name

    return tsattrs, req.url


def get_data(tsattrs, start=None, end=None, external=True, keep_timezone=False):
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
        Use Bureau of Meterology external Kiwis server. If False, use Bureau internal
        server (accessible within Bureau network only).
    keep_timezone : bool
        Keep time zone information in date parsing

    Returns
    -----------
    ts : pandas.core.series.Series
        Time series
    url : str
        Url used to query the Kiwis server
    '''
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

    if js is None or len(js[0]['data']) == 0:
        raise ValueError('Request returns no data. URL={0}'.format(req.url))

    else:
        d = pd.DataFrame(js[0]['data'], columns=['time', 'value'])

        # Discard time zone
        if keep_timezone:
            time = pd.to_datetime(d['time'])
        else:
            time = pd.to_datetime(d['time'].apply(lambda x: \
                                    re.sub('\+.*', '', x)))

        ts = d.set_index(time)['value']
        ts.name = '%s[%s]' % (tsattrs['ts_id'], tsattrs['ts_unitsymbol'])

    return ts, req.url

