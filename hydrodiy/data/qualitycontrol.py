import numpy as np
import pandas as pd

from hydrodiy.data import dutils
import c_hydrodiy_data

EPS = 1e-10

def check1d(x):
    ''' Check variable is one d only '''
    # Get numpy array data from pandas
    if hasattr(x, 'values'):
        x = x.values

    # Check dimensions
    x = np.atleast_1d(x).astype(np.float64).squeeze()
    if x.ndim > 1:
        raise ValueError('Expected 1d array, got shape {0}'.format(x.shape))

    # Check nan and infinite values
    idxok = pd.notnull(x) & np.isfinite(x)
    nok = np.sum(idxok)
    if nok == 0:
        raise ValueError('Expected some valid data in x. Got none')

    return x, idxok, nok


def islinear(data, npoints=3, tol=1e-6, thresh=0.):
    '''
    Detect linearly interpolated data

    Parameters
    -----------
    data : numpy.ndarray data
        Data series where linear interpolation is suspected
    npoints : int
        Number of points before and after current to test linearity
    tol : float
        Maximum distance between current point and linear interpolation
        to validate interpolation
    thresh : float
        Minimum threshold below which value is considered to be zero

    Returns
    -----------
    linstatus : numpy.ndarray
        Linear interpolation flag:
        * 1 = linear trend
        * 2 = constant trend

    Example
    -----------
    >>> import numpy as np
    >>> from hydrodiy.data import qualitycontrol
    >>> data = np.array([1., 2., 3., 3., 4., 5.])
    >>> qualitycontrol.islinear(data)
    array([False, True, False, False, True, False], dtype=int32)

    '''
    # Check data
    data, _, _ = check1d(data)
    islin = np.zeros(len(data), dtype=np.int32)
    npoints = int(npoints)
    tol = np.float64(tol)
    thresh = np.float64(thresh)

    if npoints<1:
        raise ValueError('Expected npoints >=1, got {0}'.format(npoints))

    if tol<1e-10:
        raise ValueError('Expected tol>1e-10, got {0:5.5e}'.format(tol))

    # Run C function
    ierr = c_hydrodiy_data.islin(thresh, tol, npoints, \
                data, islin)

    if ierr>0:
        raise ValueError('c_hydrodiy_data.islin returns {0}'.format(ierr))

    return islin



