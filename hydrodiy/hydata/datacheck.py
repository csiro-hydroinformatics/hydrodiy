import numpy as np
import c_hydata

def lindetect(data, params=[1, 1e-5]):
    ''' 
        Detect linearly interpolated data 
        
        :param np.array data: Data series where linear interpolation is suspected
        :param list params: Algorithm parameters
            params[0] : number of points before and after current point considered
            params[1] : absolute/relative tolerance
    '''
    
    # run C code via cython
    linstatus = np.zeros(len(data), np.int32)
    params = np.array(params, float)
    ierr = c_hydata.lindetect(params, data, linstatus)

    if ierr!=0:
        raise ValueError('lindetect returns %d'%ierr)

    return linstatus

