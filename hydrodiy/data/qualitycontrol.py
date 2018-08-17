import numpy as np
import pandas as pd

from hydrodiy.data import dutils
import c_hydrodiy_data


def ismisscens(x, censor=0., eps=1e-10):
    ''' Check if 1d variable is missing or censored

    Parameters
    -----------
    x : numpy.ndarray data
        1D Data series (works with pandas.Series too)
        returns an error if x is more than 1d.
    censor : float
        Censoring threshold
    eps : float
        Detection limit for censored data (i.e. x < censor + eps)

    Returns
    -----------
    icens : numpy.ndarray
        Censoring flagsLinear interpolation flag:
        * 0 = Missing data
        * 1 = Censored data
        * 2 = Valid and not censored
    '''
    # Check data for 1d
    if x.ndim > 1:
        x = x.squeeze()
    if x.ndim > 1:
        raise ValueError('Expected 1d vector, got '+\
                'x.shape = {0}'.format(x.shape))

    icens = 2*np.ones(len(x), dtype=np.int64)
    icens[pd.isnull(x) | ~np.isfinite(x)] = 0
    icens[x < censor + eps] = 1

    # Convert to pandas series if needed
    if isinstance(x, pd.Series):
        icens = pd.Series(icens, index=x.index)

    return icens


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



