import numpy as np
from scipy.linalg import toeplitz

from hydrodiy import has_c_module

def armodel_sim(params, innov, sim_mean=0., sim_ini=None):
    """ Simulate outputs from an AR model.

    If there are nan in innov, the function produces nan, but
    the internal states are kept in memory.

    Parameters
    -----------
    params : float or np.ndarray
        AR coefficients. If a float is given, the
        value is repeated n times across the time series
    innov : numpy.ndarray
        Innovation time series. [n, p] array:
        - n is the number of time steps
        - p is the number of time series to process
    sim_mean : float
        Average of output.
    sim_ini : float
        Initial condition. If None, set to sim_mean.

    Returns
    -----------
    outputs : numpy.ndarray
        Time series of AR simulations. [n, p] or [n] array
        if p=1.

    Example
    -----------
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.armodel_sim(0.95, innov1)
    >>> innov2 = sutils.armodel_residual(0.95, data)
    >>> np.allclose(innov1, innov2)
    True

    """
    has_c_module("stat")

    sim_mean = np.float64(sim_mean)
    if sim_ini is None:
        sim_ini = sim_mean
    else:
        sim_ini = np.float64(sim_ini)

    shape_innov = innov.shape
    innov = np.atleast_1d(innov).astype(np.float64)

    # set the array contiguous to work with C
    if not innov.flags["C_CONTIGUOUS"]:
        innov = np.ascontiguousarray(innov)

    # Set params
    params = np.atleast_1d(params).astype(np.float64)

    # initialise outputs
    outputs = np.zeros_like(innov)

    # Run model
    ierr = c_hydrodiy_stat.armodel_sim(sim_mean, sim_ini, params, \
                                            innov, outputs)
    if ierr!=0:
        raise ValueError("c_hydrodiy_stat.armodel_sim returns %d"%ierr)

    return np.reshape(outputs, shape_innov)


def armodel_residual(params, inputs, sim_mean=None, sim_ini=None):
    """ Compute residuals of an AR model.
    If there are nan in inputs, the function will estimate the previous
    values by applying the same AR model with a converging value
    towards sim_ini.

    Parameters
    -----------
    params : float or np.ndarray
        AR coefficient. If a float is given, the
        value is repeated n times across the time series
    inputs : numpy.ndarray
        AR1 time series. [n, p] array
        - n is the number of time steps
        - p is the number of time series to process
    sim_mean : float
        Average of output. If None set to mean(inputs)
    sim_ini : float
        Initial condition. If None, set to sim_mean.

    Returns
    -----------
    residuals : numpy.ndarray
        Time series of residuals. [n, p] array

    Example
    -----------
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.armodel_sim(0.95, innov1)
    >>> innov2 = sutils.armodel_residual(0.95, data)
    >>> np.allclose(innov1, innov2)
    True

    """
    has_c_module("stat")

    if sim_mean is None:
        sim_mean = np.nanmean(inputs).astype(np.float64)
    else:
        sim_mean = np.float64(sim_mean)

    if sim_ini is None:
        sim_ini = sim_mean
    else:
        sim_ini = np.float64(sim_ini)

    shape_inputs = inputs.shape
    inputs = np.atleast_1d(inputs).astype(np.float64)

    # set the array contiguous to work with C
    if not inputs.flags["C_CONTIGUOUS"]:
        inputs = np.ascontiguousarray(inputs)

    # Set params
    params = np.atleast_1d(params).astype(np.float64)

    # Initialise innov
    residuals = np.zeros_like(inputs)

    # Run model
    ierr = c_hydrodiy_stat.armodel_residual(sim_mean, sim_ini, \
                        params, inputs, residuals)
    if ierr!=0:
        raise ValueError("c_hydrodiy_stat.armodel_residual returns %d"%ierr)

    return np.reshape(residuals, shape_inputs)


def yule_walker(acf):
    """ Compute AR model parameter using the Yule-Walker rule.
    See https://en.wikipedia.org/wiki/Autoregressive_model#Yule%E2%80%93Walker_equations

    Code implemented as per
    http://mpastell.com/pweave/_downloads/AR_yw.html

    acf can be computed using hydrodiy.stat.sutils.acf function.

    Parameters
    -----------
    acf : numpy.ndarray
        Vector of lagged auto-correlations coefficients.
        The first element of acf should be 1.
        The size of the vector defines the order minus 1
        (e.g. len(acf) = 2 leads to an AR1 model).

    Returns
    -----------
    params : numpy.ndarray
        Parameters of the AR model.
    """
    order = len(acf)-1
    R = toeplitz(acf[:order])
    return np.dot(np.linalg.inv(R), acf[1:])
