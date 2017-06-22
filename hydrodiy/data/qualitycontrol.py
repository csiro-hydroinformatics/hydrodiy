import numpy as np

from hydrodiy.data import dutils
import c_hydrodiy_data

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
        Maximum distance between current point and linear interpolation to validate interpolation
    thresh : float
        Minimum threshold below which value is considered to be zero

    Returns
    -----------
    linstatus : numpy.ndarray
        Booleans stating if current point is interpolated or not

    Example
    -----------
    >>> import numpy as np
    >>> from hydrodiy.data import qualitycontrol
    >>> data = np.array([1., 2., 3., 3., 4., 5.])
    >>> qualitycontrol.islinear(data)
    array([False, True, False, False, True, False], dtype=int32)

    '''
    # Check data
    data = np.atleast_1d(data).astype(np.float64)
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

    # convert to bool
    islin = islin.astype(bool)

    return islin


