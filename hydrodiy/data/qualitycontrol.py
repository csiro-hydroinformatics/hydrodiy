import numpy as np


def islinear(data, npoints=1, eps=1e-5):
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
    nval = data.shape[0]
    if nval < 2*npoints:
        raise ValueError('data has less than {0} points'.format(npoints))

    d0 = data[npoints+1:]
    d1 = data[npoints:-npoints]
    d2 = data[:-(npoints+1)]

    interp = (d0+d2) * 0.5

    dist = np.abs(interp - d1)

    status = np.array([False] * len(data))

    # Set status to True for points were linear interpolation
    # is valid and either one of endpoints is non zero
    st = (dist < eps) & ((np.abs(d0)>0)|(np.abs(d2)>0))
    status[npoints:-npoints] = st

    return status


