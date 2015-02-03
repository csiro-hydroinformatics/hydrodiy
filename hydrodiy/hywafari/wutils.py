import re
import os
import json
from datetime import datetime, date
import numpy as np
import pandas as pd

import tables
from tables.exceptions import NoSuchNodeError, NodeError

from hyio import iutils

def flattens_json(jsfile):
    ''' Convert a wafari json config file into flat pandas'''
    
    config = json.load(file(jsfile))['project']
    id_all = []
    sites = dict(number=[], id=[], drainage=[], basin=[], catchment=[])
    count = 1
    for drainage in config.iterkeys():
        for basin in config[drainage]:
            for catch in config[drainage][basin]:
                id = config[drainage][basin][catch].keys()[0]
                sites['number'].append(count)
                sites['id'].append(id)
                sites['drainage'].append(drainage)
                sites['basin'].append(basin)
                sites['catchment'].append(catch) 
                count += 1

    sites = pd.DataFrame(sites, index=sites['id'])
    return sites

def has_duplicates(sites, field):
    ''' Check duplicate values '''

    nb = sites.groupby(field).apply(len)
    out = False
    ids = ''
    if np.sum(nb>1):
        out = True
        ids = ' '.join(nb[nb>1].index)

    return out, ids

def get_sites(project):

    # list of sites
    lf = iutils.find_files(project, 'report.*.json')
    sites = None
    for f in lf:
        s = flattens_json(f)
        if sites is None: sites = s
        else:   
            sites = sites.append(s[~s['id'].isin(sites['id'])])

    # Add catchments and basins
    basins, catchments = read_basin(project)
    catchments = pd.merge(catchments, basins, on = 'basin_id')
    sites = pd.merge(sites, catchments, on = 'id', how='left')

    return sites

def create_project(sites, project, model):
    
    create_projectdirs(sites, project, model)

    if not sites is None:
        create_basinjson(sites, project)
        create_simoptsjson(sites, project, model)
        create_reportjson(sites, project)


def create_projectdirs(sites, project, model):

    # Check catchment duplicates
    dp, ids = has_duplicates(sites, 'catchment')
    if dp: raise ValueError('Catchments %s occured more than once in the site list'%ids)

    # Create project folders
    F = [project, '%s/data'%project, 
       '%s/output'%project, 
        '%s/output/%s'%(project, model), 
       '%s/output/poama_m24'%project]

    for f in F:
        if not os.path.exists(f): os.mkdir(f)

    # Create site folders
    if not sites is None:

        for idx, row in sites.iterrows():
            basin = row['basin']
            catchment = row['catchment']

            # Create folders
            F = [project, '%s/data'%project, 
                '%s/data/%s'%(project, basin), 
                '%s/data/%s/meta'%(project, basin), 
                '%s/data/%s/tseries'%(project, basin),
                '%s/output'%project, 
                '%s/output/%s'%(project, model), 
                '%s/output/%s/%s'%(project, model, basin), 
                '%s/output/%s/%s/%s'%(project, model, basin, catchment),
                '%s/output/%s/%s/%s/etc'%(project, model, basin, catchment), 
                '%s/output/%s/%s/%s/out'%(project, model, basin, catchment),
                '%s/output/poama_m24'%project, 
                '%s/output/poama_m24/%s'%(project, basin), 
                '%s/output/poama_m24/%s/%s'%(project, basin, catchment),
                '%s/output/poama_m24/%s/%s/out'%(project, basin, catchment)]

            for f in F:
                if not os.path.exists(f): os.mkdir(f)

def create_basinjson(sites, project):

    # Check id duplicates
    dp, ids = has_duplicates(sites, 'id')
    if dp: raise ValueError('Ids %s occured more than once in the site list'%ids)

    for idx, row in sites.iterrows():

        basin = row['basin']
        id = row['id']
        name = row['name']
        desc = row['description']
        area = row['area']

        fb = '%s/data/%s/meta/basin.json'%(project,basin)
        if os.path.exists(fb):
            fbb = open(fb, 'r')
            txt = fbb.readlines()
            fbb.close()
            basin_data = json.loads(' '.join(txt))

        else:
            basin_data = {'conventionName':'basin.ehp.bom.gov.au',
                'conventionVersion':'1.0', 
                'name':basin, 'description':'%s basin'%re.sub('_', ' ', basin), 
                'area': -999., 'areaUnits':'km^2', 
                'centroidCoordinates':[-999., -999.], 'catchment':[]}

        catch_data = {'ID':'%s'%id, 'AWRC':'%s'%id, 
            'name':name, 'description':desc,
            'area':area, 'areaUnits':'km^2'}
        basin_data['catchment'].append(catch_data)

        txt = json.dumps(basin_data, indent=4)
        fbb = open(fb, 'w')
        fbb.writelines(txt)
        fbb.close()

def create_simoptsjson(sites, project, model):

    # Check id duplicates
    dp, ids = has_duplicates(sites, 'id')
    if dp: raise ValueError('Ids %s occured more than once in the site list'%ids)

    # Create common simopts
    fb = '%s/output/%s/simopts.json'%(project, model)
    simopts_data = {'conventionName':'common_simopts.ehp.bom.gov.au',
        'conventionVersion':'1.0', 
        'conventionDescription':'common simulation options', 
        'modelName': model, 
        "crossValidation": {
            "startDate": "1980-04-01",
            "endDate": "2008-12-01",
        }}
    txt = json.dumps(simopts_data, indent=4)
    fbb = open(fb, 'w')
    fbb.writelines(txt)
    fbb.close()

    # Create simopts for individual sites
    for idx, row in sites.iterrows():

        basin = row['basin']
        catchment = row['catchment']
        id = row['id']

        fb = '%s/output/%s/%s/%s/etc/simopts.json'%(project, model, basin, catchment)
        simopts_data = {'conventionName':'simopts.ehp.bom.gov.au',
            'conventionVersion':'1.0', 
            'conventionDescription':'%s simulation options'%model, 
            'modelName': model, 'comment':'',
            "parameter": {
                "startDate": "1980-01-01",
                "endDate": "2008-12-01",
                "warmupDate": "1975-01-01",
                "paramStartDate": "1980-01-01",
                "paramWarmupDate": "1970-01-01",
                "leaveOut": 5,
                "numberOfSamples": 6200,
                "useBateaErrorModel": 'true',
                "infillToStartDate": 'true'
            },
            "variable": {
                "Q": {
                    "ID": '%s'%id,
                    "type": "streamflow",
                    "source": "station",
                    "frequency": "daily",
                    "lagTime": "P1M",
                    "integrationTime": "P3M"
                },
                "P": {
                    "ID": '%s'%id,
                    "type": "precipitation",
                    "source": "station",
                    "frequency": "daily",
                    "lagTime": "P1M",
                    "integrationTime": "P3M"
                },
                "P_P24": {
                    "ID": '%s'%id,
                    "type": "precipitation",
                    "source": "downscaled_poama_2.4",
                    "version": "m24",
                    "frequency": "daily",
                    "lagTime": "P1M",
                    "integrationTime": "P3M"
                },
                "PET": {
                    "ID": '%s'%id,
                    "type": "potential evaporation",
                    "source": "station",
                    "frequency": "daily",
                    "lagTime": "P1M",
                    "integrationTime": "P3M"
                }
            },
            "predictor": [
                [
                    {
                        "variable": "P_P24",
                        "integrationTime": "P3M"
                    },
                    {
                        "variable": "P",
                        "integrationTime": "P3M"
                    },
                    {
                        "variable": "PET",
                        "integrationTime": "P3M"
                    }
                ]
            ],
            "predictand": [
                {
                    "lagTime": "P1M",
                    "leadTime": "P1M",
                    "source": "station",
                    "frequency": "daily",
                    "variable": "Q",
                    "type": "streamflow",
                    "ID": '%s'%id,
                    "integrationTime": "P3M"
                }
            ],
            "calibrate_predictor": [
                {
                    "variable": "P",
                    "integrationTime": "P3M"
                },
                {
                    "variable": "PET",
                    "integrationTime": "P3M"
                }
            ],
            "calibrate_predictand": [
                {
                    "variable": "Q",
                    "integrationTime": "P3M"
                }
            ]
        }

        txt = json.dumps(simopts_data, indent=4)
        fbb = open(fb, 'w')
        fbb.writelines(txt)
        fbb.close()

def create_reportjson(sites, project, jsonfile='report.json'):

    # Check id duplicates
    dp, ids = has_duplicates(sites, 'id')
    if dp: raise ValueError('Ids %s occured more than once in the site list'%ids)

    fr = '%s/%s'%(project, jsonfile) 
    if os.path.exists(fr):
        frr = open(fr, 'r')
        txt = frr.readlines()
        frr.close()
        report_data = json.loads(' '.join(txt))
    
    else:
        report_data = {'conventionName':'report.ehp.bom.gov.au',
            'conventionVersion':'1.0', 
            'conventionDescription':'1.0', 
            'startYear':2011,
            'server':{ 'url':'wafari-gb.bom.gov.au', 
                    'directory':'/var/www/html/water/reg/ehp/',
                    'user':'ehpop'},
            'project':{}}

    for idx, row in sites.iterrows():

        drainage = row['drainage']
        basin = row['basin']
        catchment = row['catchment']
        id = row['id']

        if drainage in report_data['project']:

            if basin in report_data['project'][drainage]:

                if catchment in report_data['project'][drainage][basin]:

                    if id in report_data['project'][drainage][basin][catchment]:
                        pass
                    else:
                        report_data['project'][drainage][basin][catchment][id] = {}

                else:
                    report_data['project'][drainage][basin][catchment] = {id:{}}
                    
            else:
                report_data['project'][drainage][basin] = {catchment:{id:{}}}
                
        else:
            drainage_data = {basin: {catchment: {id:{}}}}
            report_data['project'][drainage] = drainage_data

    txt = json.dumps(report_data, indent=4)
    frr = open(fr, 'w')
    frr.writelines(txt)
    frr.close()


def read_basin(PROJECT):
    ''' read basin json file from project '''

    lf = iutils.find_files('%s/data'%PROJECT,'.*basin.json')
    basins = []
    catchments = []
    nb = 1

    if len(lf)>0:
        for f in lf:
            fj = open(f, 'r')
            txt = fj.readlines()
            fj.close() 
            js = json.loads(' '.join(txt))

            # Extract basins
            b = {'basin_name':'', 'basin_area':0., 
                    'basin_description':'', 'basin_id':'B%3.3d'%nb,
                    'basin_centroid_long':0., 
                    'basin_centroid_lat':0}

            for k in ['name', 'area', 'description']:
                if k in js: b['basin_%s'%k] = js[k]        

            if 'centroidCoordinates' in js:
                b['basin_centroid_long'] = js['centroidCoordinates'][1]
                b['basin_centroid_lat'] = js['centroidCoordinates'][0]

            basins.append(b)
            nb += 1

            # Extract catchment
            if 'catchment' in js:
                for catch in js['catchment']:
                    catchments.append(catch)
                    catchments[-1]['basin_id'] = b['basin_id']
    
    basins = pd.DataFrame(basins)
    basins = basins.drop_duplicates(subset=['basin_name'])

    catchments = pd.DataFrame(catchments)
    catchments = catchments.rename(columns={'ID':'id'})
    catchments = catchments.drop_duplicates(subset=['id', 'name'])

    return basins, catchments

def read_fc(h5file, station_id, variable='STREAMFLOW'):
    '''  
    
    reads simulation data from a forecast.hdf5 file

    '''
    simulations = None
    with tables.openFile(h5file, mode='r') as h5:
        simulations = h5.get_node('/data/forecast/%s.%s'%(station_id, variable)).read()
    
    return simulations


def readsim_xvalidate(h5file, station_id, variable='STREAMFLOW'):
    '''  
    
    reads simulation data from a xvalidate.hdf5 file
    handles both formats (wafari v1 and v2)

    '''
    simulations = None
    with tables.openFile(h5file, mode='r') as h5:

        for nd in h5.walk_nodes('/data/forecast', 'Array'):
           
            # Search id in node path
            se =  re.search('/(.|)%s(|[A-Z]).*simulation'%station_id, 
                                        nd._v_pathname)

            # Search variable name in node title attribute
            se2  = re.search(variable, nd.title)

            # proceeds if both searches returns something
            if (se is not None) & (se2 is not None):
                # Get simulation values
                value = nd.read()

                # Find month
                se2 = re.search('M[\\d]{2}D[\\d]{2}',
                        nd._v_pathname)
                month = re.sub('M|D', '-', se2.group(0))   

                # Find year
                ndy = re.sub(r'simulation', 'year', 
                        nd._v_pathname)
                years = h5.get_node(ndy).read()

                # Find obs
                ndo = re.sub(r'simulation', 'observation', 
                        nd._v_pathname)
                obs = h5.get_node(ndo).read()

                # Build index
                index = [datetime.strptime('%s%s'%(y ,month), 
                                    '%Y-%m-%d') for y in years]

                # Build dataframe
                data = pd.DataFrame(value, index=index)
                data.columns = ['Ens%4.4d'%i for i in 
                                    range(1, data.shape[1]+1)] 
                data['obs'] = obs

                if simulations is None:
                    simulations = data
                else:
                    simulations = pd.concat([simulations, data])

    if not simulations is None:
        simulations = simulations.sort()

    return simulations

def readscores_xvalidate(h5file, station_id):
    '''  
    
    reads scores data from a xvalidate.hdf5 file
    handles both formats (wafari v1 and v2)

    '''
    scores = None
    with tables.openFile(h5file, mode='r') as h5:

        for nd in h5.walk_nodes('/data/skillScore', 'Array'):
           
            # Search id in node path
            se =  re.search('/(.|)%s(|[A-Z])\.STREAMFLOW.*(CRPS|RMSE|RMSEP)$'%station_id, 
                                        nd._v_pathname)
            # proceeds if both searches returns something
            if se is not None:
                # Get values
                values = nd.read()

                # Find score name
                se2 = re.search('(CRPS|RMSE|RMSEP)$', nd._v_pathname)
                scname = se2.group(0)   

                # Find month
                se2 = re.search('M[\\d]{2}', nd._v_pathname)
                month = int(re.sub('M', '', se2.group(0))) 

                # Build dataframe
                data = pd.DataFrame({'value':values})
                data['score_name'] = scname
                data['type'] = ['skill', 'score', 'climatology']
                data['month'] = month
                data['node'] = nd._v_pathname

                if scores is None:
                    scores = data
                else:
                    scores = pd.concat([scores, data])

    if scores is None:
        raise ValueError('Cannot extract scores (=None)')

    if scores.shape[0]!=108:
        raise ValueError('scores does not have 108 lines (=%d)' % scores.shape[0])

    return scores

def readrefs_xvalidate(h5file, station_id, variable='STREAMFLOW'):
    '''  
    
    reads refs data from a xvalidate.hdf5 file
    handles both formats (wafari v1 and v2)

    '''
    reference = None
    with tables.openFile(h5file, mode='r') as h5:

        for nd in h5.walk_nodes('/data/forecast', 'Array'):
           
            # Search id in node path
            se =  re.search('/(.|)%s(|[A-Z]).*reference'%station_id, 
                                        nd._v_pathname)

            # Search variable name in node title attribute
            se2  = re.search(variable, nd.title)

            # proceeds if both searches returns something
            if (se is not None) & (se2 is not None):

                # Get simulation values
                refs = nd.read()

                # Find month
                se2 = re.search('M[\\d]{2}', nd._v_pathname)
                month = int(re.sub('M', '', se2.group(0)))   

                # Build dataframe
                data = pd.DataFrame(refs).T
                data.columns = ['Ens%4.4d'%i for i in 
                                    range(1, data.shape[1]+1)] 
                data['month'] = month
                data = data.set_index('month')

                if reference is None:
                    reference = data
                else:
                    reference = pd.concat([reference, data])
             
    if not reference is None:
        reference = reference.sort()

    return reference

def read_obs(h5file, station_id, variable, frequency):

    # check inputs
    if not variable in ['STREAMFLOW', 'PRECIPITATION', 'POTENTIAL_EVAPORATION']:
        raise ValueError('variable %s cannot be used to read obs file'%variable)

    if variable=='STREAMFLOW' and not h5file.endswith('streamflow.hdf5'):
        raise ValueError('File must be named streamflow.hdf5 for STREAMFLOW variable')
        
    if variable.startswith('P') and not h5file.endswith('hydromet.hdf5'):
        raise ValueError('File must be named hydromet.hdf5 for PRECIPITATION and POTENTIAL_EVAPORATION variables')

    if not frequency in ['daily', 'monthly']:
        raise ValueError('frequency %s cannot be used to read obs file'%frequency)

    station_id = str(station_id)

    data = None
    with tables.openFile(h5file, mode='r') as h5:
        data = h5.get_node('/data/%s/%s/%s'%(frequency,variable, station_id)).read()

    data = pd.DataFrame(data)
    data['isotime'] = pd.to_datetime(data['isotime'])
    data = data.set_index('isotime')

    return data

def create_obs(h5file, station_id, variable, obs):

    # check inputs
    if not variable in ['STREAMFLOW', 'PRECIPITATION', 'POTENTIAL_EVAPORATION']:
        raise ValueError('variable %s cannot be used to create obs file'%variable)

    if variable=='STREAMFLOW' and not h5file.endswith('streamflow.hdf5'):
        raise ValueError('File must be named streamflow.hdf5 for STREAMFLOW variable')
        
    if variable.startswith('P') and not h5file.endswith('hydromet.hdf5'):
        raise ValueError('File must be named hydromet.hdf5 for PRECIPITATION and POTENTIAL_EVAPORATION variables')

    station_id = str(station_id)

    # Check file exists
    h5mode = 'w'
    if os.path.exists(h5file):
        h5mode = 'a'

    # Populate data file
    with tables.openFile(h5file, mode=h5mode) as h5:

        # Meta data
        try:
            meta_data = h5.createArray(
                "/", "meta", 0, "meta data. see its attributes."
            )
            meta_data.attrs.conventionDescription = 'measured hydrometeorological data HDF5 file format'
            meta_data.attrs.conventionName = 'hydromet.ehp.bom.gov.au'
            if variable =='STREAMFLOW': 
                meta_data.attrs.conventionDescription = 'measured streamflow data HDF5 file format'
                meta_data.attrs.conventionName = 'streamflow.ehp.bom.gov.au'
            meta_data.attrs.dataOwner = "Bureau of Meteorology"
            meta_data.attrs.dataProvider = "Bureau of Meteorology"
            meta_data.attrs.creationDate = datetime.now().isoformat()
            meta_data.attrs.recentRevisionDate = datetime.now().isoformat()
            meta_data.attrs.referenceTime = date.fromordinal(1).isoformat()
            meta_data.attrs.revisionHistory = (date.today().isoformat() +
                                              ": created the file.")
            meta_data.attrs.comment = "File created with hydrodiy library"
        except NodeError:
            pass

        # Groups
        for gr in ['/data', '/data/daily', '/data/monthly', '/data/daily/%s'%variable,
                '/data/monthly/%s'%variable]:
            grn = re.sub('.*/', '', gr)
            grw = re.sub(grn, '', gr)
            try:
                h5.createGroup(grw, grn, '%s group'%grn) 
            except NodeError: 
                pass

        # Data Table class
        class TabDesc(tables.IsDescription):
            time = tables.Int32Col(dflt=0, pos=0)
            isotime = tables.StringCol(10)
            value = tables.Float64Col(dflt=np.nan, pos=1)
            flag = tables.Int32Col(dflt=1, pos=2)

        # Create tables and populate at daily and monthly time step
        for frequency in ['daily', 'monthly']:

            # Create table
            try:
                h5.removeNode(h5.getNode('/data/%s/%s' % (frequency, variable)),
                       station_id, recursive=True)
            except NoSuchNodeError:
                    pass

            varTab = h5.createTable(h5.getNode('/data/%s/%s' % (frequency, variable)),
                station_id, TabDesc, station_id)

            varTab.attrs.description= station_id
            varTab.attrs.fillValue = np.nan
            varTab.attrs.group = ""
            varTab.attrs.method = ""
           
            # compute monthly aggregation is required
            data = obs
            if frequency=='monthly': data = obs.resample('MS', how='sum')

            # insert data
            indexRow = varTab.row
            for idx, value in data.iteritems():
                indexRow["time"] = idx.to_datetime().toordinal()
                indexRow["isotime"] = idx.to_datetime().isoformat()
                indexRow["value"] = value
                indexRow["flag"] = 1
                indexRow.append()
                                                                      
        # Store revision date
        h5.root.meta.attrs.recentRevisionDate = \
            datetime.now().isoformat()

        h5.flush()

def create_poama_hindcast(h5file, forecasts, nleadtime=92, timestep='days'):

    # check inputs
    if not h5file.endswith('hindcast.hdf5'):
        raise ValueError('File must be named hindcast.hdf5')

    # dimensions
    dates = np.unique(forecasts['forecast_date'])
    ndates = len(dates)

    ens = np.unique(forecasts['iens'])
    nens = len(ens)

    # Populate data file
    with tables.openFile(h5file, mode='a') as h5:

        # Meta data
        try:
            meta_data = h5.createArray(
                "/", "meta", 0, "meta data. see its attributes."
            )
            meta_data.attrs.conventionDescription = 'calculated calibration and simulation results for cross validation in HDF5 file'
            meta_data.attrs.conventionName = 'simulation.ehp.bom.gov.au'
            meta_data.attrs.conventionVersion = '1.2'
            meta_data.attrs.dataOwner = "Bureau of Meteorology"
            meta_data.attrs.dataProvider = "Bureau of Meteorology"
            meta_data.attrs.creationDate = datetime.now().isoformat()
            meta_data.attrs.recentRevisionDate = datetime.now().isoformat()
            meta_data.attrs.referenceTime = date.fromordinal(1).isoformat()
            meta_data.attrs.revisionHistory = (date.today().isoformat() +
                                              ": created the file.")
            meta_data.attrs.comment = "File created with hydrodiy library"
        except NodeError:
            pass

        # Groups
        for gr in ['/data', '/data/YALL']:
            grn = re.sub('.*/', '', gr)
            grw = re.sub(grn, '', gr)
            try:
                h5.createGroup(grw, grn, '%s group'%grn) 
            except NodeError: 
                pass

        # Data Table description
        table_description = {
            'date': tables.Int32Col(dflt=0, pos=0),
            'isodate': tables.StringCol(10, pos=1),
            'parameter': tables.Int32Col(dflt=0, pos=2),
            'ensemble': tables.Int32Col(dflt=0, pos=2),
        }
        for i in xrange(nleadtime):
            table_description['lead%d' % i] = tables.Float64Col(dflt=0, pos=3)

        # Create table
        try:
            h5.removeNode(h5.getNode('/data/YALL/simulation'))
        except NoSuchNodeError:
                pass

        varTab = h5.createTable('/data/YALL', 'simulation', table_description)
        varTab.attrs.fillValue = np.nan
        varTab.attrs.ndates = ndates
        varTab.attrs.nparams = 1
        varTab.attrs.nvalues = nleadtime
        varTab.attrs.nens = nens
        varTab.attrs.timestep = timestep
        varTab.attrs.year = 'YALL'
        varTab.attrs.group = ""
        varTab.attrs.method = ""
        
        # insert data
        index_row = varTab.row
        count = 0
        for idx, row in forecasts.iterrows():
            count += 1
            if count % 500 ==0: print('.. appending row %7d/%7d ..'%(count, forecasts.shape[0]))

            index_row['date'] = row['forecast_date'].toordinal()
            index_row['isodate'] = row['forecast_date'].isoformat()
            index_row['parameter'] = 0
            index_row['ensemble'] = row['iens']
            for j in range(nleadtime):
                index_row['lead%d' % j] = row['lead%2.2d'%j]
                
            index_row.append()
                                                                      
        # Store revision date
        h5.root.meta.attrs.recentRevisionDate = \
            datetime.now().isoformat()

        h5.flush()

def create_gr4j_hindcast(h5file, states, parameters):

    # check inputs
    if not h5file.endswith('hindcast.hdf5'):
        raise ValueError('File must be named hindcast.hdf5')

    # dimensions
    dates = np.unique(forecasts['forecast_date'])
    ndates = len(dates)

    ens = np.unique(forecasts['iens'])
    nens = len(ens)

    # Populate data file
    with tables.openFile(h5file, mode='a') as h5:

        # Meta data
        try:
            meta_data = h5.createArray(
                "/", "meta", 0, "meta data. see its attributes."
            )
            meta_data.attrs.conventionDescription = 'calculated calibration and simulation results for cross validation in HDF5 file'
            meta_data.attrs.conventionName = 'simulation.ehp.bom.gov.au'
            meta_data.attrs.conventionVersion = '1.2'
            meta_data.attrs.dataOwner = "Bureau of Meteorology"
            meta_data.attrs.dataProvider = "Bureau of Meteorology"
            meta_data.attrs.creationDate = datetime.now().isoformat()
            meta_data.attrs.recentRevisionDate = datetime.now().isoformat()
            meta_data.attrs.referenceTime = date.fromordinal(1).isoformat()
            meta_data.attrs.revisionHistory = (date.today().isoformat() +
                                              ": created the file.")
            meta_data.attrs.comment = "File created with hydrodiy library"
        except NodeError:
            pass

        # Groups
        for gr in ['/data', '/data/YALL']:
            grn = re.sub('.*/', '', gr)
            grw = re.sub(grn, '', gr)
            try:
                h5.createGroup(grw, grn, '%s group'%grn) 
            except NodeError: 
                pass

        # Data Table description
        table_description = {
            'date': tables.Int32Col(dflt=0, pos=0),
            'isodate': tables.StringCol(10, pos=1),
            'parameter': tables.Int32Col(dflt=0, pos=2),
            'ensemble': tables.Int32Col(dflt=0, pos=2),
        }
        for i in xrange(nleadtime):
            table_description['lead%d' % i] = tables.Float64Col(dflt=0, pos=3)

        # Create table
        try:
            h5.removeNode(h5.getNode('/data/YALL/simulation'))
        except NoSuchNodeError:
                pass

        varTab = h5.createTable('/data/YALL', 'simulation', table_description)
        varTab.attrs.fillValue = np.nan
        varTab.attrs.ndates = ndates
        varTab.attrs.nparams = 1
        varTab.attrs.nvalues = nleadtime
        varTab.attrs.nens = nens
        varTab.attrs.timestep = timestep
        varTab.attrs.year = 'YALL'
        varTab.attrs.group = ""
        varTab.attrs.method = ""
        
        # insert data
        index_row = varTab.row
        count = 0
        for idx, row in forecasts.iterrows():
            count += 1
            if count % 500 ==0: print('.. appending row %7d/%7d ..'%(count, forecasts.shape[0]))

            index_row['date'] = row['forecast_date'].toordinal()
            index_row['isodate'] = row['forecast_date'].isoformat()
            index_row['parameter'] = 0
            index_row['ensemble'] = row['iens']
            for j in range(nleadtime):
                index_row['lead%d' % j] = row['lead%2.2d'%j]
                
            index_row.append()
                                                                      
        # Store revision date
        h5.root.meta.attrs.recentRevisionDate = \
            datetime.now().isoformat()

        h5.flush()


