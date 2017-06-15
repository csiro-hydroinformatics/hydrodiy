import numpy as np


def islinear(data, npoints=1, tol=None, thresh=None):
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

    data = np.atleast_1d(data)

    # Compute min threhsold
    if thresh is None:
        thresh = np.nanmin(data[data>0])/10.

    # Compute tol as the min of the diff between two points divided by 10
    if tol is None:
        diff = np.abs(np.diff(data))
        diff = diff[diff>thresh]
        tol = np.nanmin(diff)/10.

    nval = data.shape[0]
    if nval < 2*npoints+2:
        raise ValueError('data has less than {0} points'.format(2*npoints+2))

    # Lag data
    lagged = np.nan * np.zeros((data.shape[0], 2*npoints+1))
    lags = np.arange(-npoints, npoints+1)
    for ilag, lag in enumerate(lags):
        v = np.roll(data, lag)
        if lag<0:
            v[lag:] = np.nan
        elif lag>0:
            v[:lag] = np.nan
        lagged[:, ilag] = v

    # Compute linear interpolation
    w = np.linspace(0, 1, 2*npoints+1)[None, :]
    interp = lagged[:, 0][:, None]*(1-w) + lagged[:, -1][:, None]*w

    # Compute distance
    dist = np.abs(interp[:, npoints]-lagged[:, npoints])

    # Set status
    status = (dist < tol) & ~np.isnan(dist) & (data>thresh)

    return status


