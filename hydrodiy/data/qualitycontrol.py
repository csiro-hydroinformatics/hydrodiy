import numpy as np


def islinear(data, npoints=1, eps=None, minthreshold=None):
    '''
    Detect linearly interpolated data

    Parameters
    -----------
    data : numpy.ndarray data
        Data series where linear interpolation is suspected
    npoints : int
        Number of points before and after current to test linearity
    eps : float
        Maximum distance between current point and linear interpolation to validate interpolation
    minthreshold : float
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

    data = np.atleast_1d(data)

    # Compute min threhsold
    if minthreshold is None:
        minthreshold = np.nanmin(data[data>0])/10.

    # Compute eps as the min of the diff between two points divided by 10
    if eps is None:
        diff = np.abs(np.diff(data))
        diff = diff[diff>0]
        eps = np.nanmin(diff)/10.

    nval = data.shape[0]
    if nval < 2*npoints+2:
        raise ValueError('data has less than {0} points'.format(2*npoints+2))

    nvall = nval-2*npoints
    lagged = np.zeros((nvall, 2*npoints+1))
    interp = lagged.copy()
    for k in range(2*npoints+1):
        lagged[:, k] = np.roll(data, k)[2*npoints:]
        interp[:, k] = np.ones(nvall) *  float(k)/(npoints+1)

    lag0 = np.repeat(lagged[:, 0].reshape((nvall, 1)), 2*npoints+1, axis=1)
    lag1 = np.repeat(lagged[:, -1].reshape((nvall, 1)), 2*npoints+1, axis=1)
    interp = (1-interp) * lag0 + interp * lag1

    dist = np.max(np.abs(interp[:, 1:-1] - lagged[:, 1:-1]), axis=1)

    # Set status to True for points were linear interpolation
    # is valid and either one of endpoints is non zero
    st = (dist < eps) & \
        ((np.abs(lag0[:, 0])>minthreshold)|(np.abs(lag1[:, -1])>minthreshold))

    status = np.array([False] * len(data))
    status[npoints:-npoints] = st

    return status


