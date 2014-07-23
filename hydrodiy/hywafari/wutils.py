import re
import json
from datetime import datetime
import numpy as np
import tables
import pandas as pd

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


def read_xvalidate(h5file, station_id, variable='STREAMFLOW'):
    '''  
    
    reads simulation data from a xvalidate.hdf5 file
    handles both formats (wafari v1 and v2)

    '''
    simulations = None
    with tables.openFile(h5file, mode='r') as h5:

        for nd in h5.walk_nodes('/data/forecast', 'Array'):
           
            # Search id in node path
            se =  re.search('/(.|)%s.*simulation'%station_id, 
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

