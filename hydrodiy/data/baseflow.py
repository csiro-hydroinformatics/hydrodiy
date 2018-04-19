
import numpy as np
import pandas as pd

import c_hydrodiy_data

def eckhardt(flow, thresh=0.95, tau=20, BFI_max=0.8, timestep_type=1):
    ''' Compute the baseflow component based on Eckhardt algorithm
    Eckhardt K. (2005) How to construct recursive digital filters for baseflow separation. Hydrological processes 19:507-515.

    C code was translated from R code provided by
    Jose Manuel Tunqui Neira, IRSTEA, 2018 (jose.tunqui@irstea.fr)

    Parameters
    -----------
    flow : numpy.array
        Streamflow data
    thresh : float
        Percentage from which the base flow should be considered as total flow
    tau : float
        Characteristic drainage timescale (hours)
    BFI_max : float
        See Eckhardt (2005)
    timestep_type : int
        Type of time step: 0=hourly, 1=daily

    Returns
    -----------
    bflow : numpy.array
        Baseflow time series

    Example
    -----------
    >>> import numpy as np
    >>> q = np.random.uniform(0, 100, size=1000)
    >>> baseflow.baseflow(q)

    '''
    # run C code via cython
    thresh = np.float64(thresh)
    tau = np.float64(tau)
    BFI_max = np.float64(BFI_max)
    timestep_type = np.int32(timestep_type)

    flow = np.array(flow).astype(np.float64)
    bflow = np.zeros(len(flow), np.float64)

    ierr = c_hydrodiy_data.eckhardt(timestep_type, \
                thresh, tau, BFI_max, flow, bflow)

    if ierr!=0:
        raise ValueError('c_hydata.eckhardt returns %d'%ierr)

    return bflow
