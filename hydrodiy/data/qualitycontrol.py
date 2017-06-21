import numpy as np

from hydrodiy.data import dutils

def islinear(data, npoints=3, tol=None, thresh=None):
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
    data = np.atleast_1d(data)
    npoints = int(npoints)

    if npoints<1:
        raise ValueError('Expected npoints >0, got {0}'.format(npoints))

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
        raise ValueError(('Given npoints={0}, expected data of ' + \
                'length at least {1}, got {2}').format(npoints, \
                2*npoints+2, nval))

    # Compute distance with linear interpolation
    # between lag -1 and lag +1
    interp = (dutils.lag(data, -1)+dutils.lag(data, 1))/2
    dist = np.abs(interp-data)

    # Set linear status for one point
    islin = ((dist < tol) & ~np.isnan(dist) & (data>thresh)).astype(float)

    # Check status before and after current point
    if npoints > 1:
        islin_pos = np.zeros(list(data.shape)+[npoints+1])
        islin_neg = np.zeros(list(data.shape)+[npoints+1])
        for il, l in enumerate(range(npoints+1)):
            islin_neg[:, il] = dutils.lag(islin, -l)
            islin_pos[:, il] = dutils.lag(islin, l)

        ndim = islin_pos.ndim
        islin = np.all(islin_pos>0, axis=ndim-1) | \
                np.all(islin_neg>0, axis=ndim-1)

    return islin


