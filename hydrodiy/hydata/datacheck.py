import numpy as np
import c_hydata

def lindetect(data, params=[1, 1e-5]):
    ''' 
    Detect linearly interpolated data 

    Parameters
    -----------
    data : numpy.ndarray data
        Data series where linear interpolation is suspected
    params : list 
        Algorithm parameters
        params[0] number of points before and after current point considered
        params[1] absolute/relative tolerance

    Returns
    -----------
    linstatus : numpy.ndarray
        Data status as per linear interpolation
        0
        1 

    Example
    -----------
    >>> import numpy as np
    >>> from hydata import datacheck
    >>> data = np.array([1., 2., 3., 3., 4., 5.])
    >>> datacheck.lindetect(data)
    array([0, 1, 0, 0, 1, 0], dtype=int32) 

    '''
    
    # run C code via cython
    linstatus = np.zeros(len(data), np.int32)
    params = np.array(params, float)
    ierr = c_hydata.lindetect(params, data, linstatus)

    if ierr!=0:
        raise ValueError('lindetect returns %d'%ierr)

    return linstatus

