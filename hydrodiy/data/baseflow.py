
import numpy as np
import pandas as pd

import c_hydrodiydev_data

def baseflow(inputs, params, method=1):
    ''' Compute baseflow time series using the algorithms defined
    by Chapman (1999)

    Parameters
    -----------
    inputs : numpy.array
        Streamflow data
    params : list
        Algorightm parameters. Has 1 item for method 1, 2 for method
        2 and 3 for method 3.
    method : int
        Method selected. 1=Chapman, 2=Boughton, 3=IHACRES

    Returns
    -----------
    outputs : numpy.array
        Baseflow time series

    Example
    -----------
    >>> import numpy as np
    >>> q = np.random.uniform(0, 100, size=1000)
    >>> baseflow.baseflow(q, [0.99])
    'A04567'

    '''

    # Check params length
    if method >= 2 and len(params)<2:
        raise ValueError('method=%d and len(params)<2' % method)

    if method == 3 and len(params)<3:
        raise ValueError('method=%d and len(params)<3' % method)

    # run C code via cython
    method = int(method)
    outputs = np.zeros(len(inputs), np.float64)
    params = np.array(params, float)

    ierr = c_hydrodiydev_data.baseflow(method, params, inputs, outputs)

    if ierr!=0:
        raise ValueError('c_hydata.baseflow returns %d'%ierr)

    return outputs
