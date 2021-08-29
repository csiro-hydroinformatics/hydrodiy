import numpy as np
import pandas as pd

from hydrodiy.data import dutils
from hydrodiy import has_c_module

if has_c_module("data", False):
    import c_hydrodiy_data

def ismisscens(x, censor=0., eps=1e-10):
    """ Check if 1d variable is missing or censored

    Parameters
    -----------
    x : numpy.ndarray data
        1D or 2d Data series (works with pandas.Series too)
        returns an error if x is more than 1d.
    censor : float
        Censoring threshold
    eps : float
        Detection limit for censored data (i.e. censored = x < censor + eps)

    Returns
    -----------
    icens : numpy.ndarray
        Censoring flags:
        * 3^i+0 = Missing data
        * 3^i+1 = Censored data
        * 3^i+2 = Valid and not censored
        where i is the column number
    """
    # Check data dimensions
    ndim = x.ndim
    if ndim > 2:
        raise ValueError("Expected 1d or 2d vector, got "+\
                "x.shape = {0}".format(x.shape))

    # Get dimensions
    if ndim == 1:
        nval = x.shape[0]
        ncols = 1
    else:
        nval, ncols = x.shape

    # Compute censoring flags
    icens = np.zeros((nval, ncols), dtype=np.int64)
    for i in range(ncols):
        # Select data
        if ndim > 1:
            u = x[:, i]
        else:
            u = x

        # find censoring flags for column
        icens[u < censor + eps, i] = 1
        icens[u >= censor + eps, i] = 2
        icens[:, i] *= 3**i

    # Aggregate
    if ndim > 1:
        icens = np.sum(icens, axis=1)
    else:
        icens = icens.squeeze()

    return icens


def islinear(data, npoints=3, tol=1e-6, thresh=0.):
    """
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

    """
    has_c_module("data")

    # Check data
    if data.ndim > 1:
        raise ValueError("Expected data as 1d vector, got "+\
            "data.shape={0}".format(data.shape))

    islin = np.zeros(len(data), dtype=np.int32)
    npoints = int(npoints)
    tol = np.float64(tol)
    thresh = np.float64(thresh)

    if npoints<1:
        raise ValueError("Expected npoints >=1, got {0}".format(npoints))

    if tol<1e-10:
        raise ValueError("Expected tol>1e-10, got {0:5.5e}".format(tol))

    # Run C function
    ierr = c_hydrodiy_data.islin(thresh, tol, npoints, \
                data, islin)

    if ierr>0:
        raise ValueError("c_hydrodiy_data.islin returns {0}".format(ierr))

    return islin



