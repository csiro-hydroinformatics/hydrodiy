import re
import json
from datetime import datetime
import numpy as np
import tables
import pandas as pd

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
    catchments = pd.DataFrame(catchments)
    catchments = catchments.rename(columns={'ID':'id'})

    return basins, catchments

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

    if not scores is None:
        assert scores.shape[0] == 108

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

