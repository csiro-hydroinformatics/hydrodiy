import re
import math
import numpy as np
import pandas as pd

from scipy.stats import norm

# Try to import C code
HAS_C_STAT_MODULE = True
try:
    import c_hydrodiy_stat
except ImportError:
    HAS_C_STAT_MODULE = False


def armodel_sim(params, innov, simini=0., fillnan=False):
    ''' Simulate outputs from an AR model.

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
    simini : float
        Initial condition
    fillnan : bool
        Infill nan values with simini or not.

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

    '''
    if not HAS_C_STAT_MODULE:
        raise ValueError('C module c_hydrodiy_stat is not available, '+\
                'please run python setup.py build')

    simini = np.float64(simini)

    shape_innov = innov.shape
    innov = np.atleast_2d(innov).astype(np.float64)
    fillnan = np.int32(fillnan)

    # Transpose 1d array
    if innov.shape[0] == 1:
        innov = innov.T

    shape = innov.shape

    # set the array contiguous to work with C
    if not innov.flags['C_CONTIGUOUS']:
        innov = np.ascontiguousarray(innov)

    # Set params
    params = np.atleast_1d(params).astype(np.float64)

    # initialise outputs
    outputs = np.zeros(shape, np.float64)

    # Run model
    ierr = c_hydrodiy_stat.armodel_sim(simini, fillnan, params, innov, outputs)
    if ierr!=0:
        raise ValueError('c_hydrodiy_stat.armodel_sim returns %d'%ierr)

    return np.reshape(outputs, shape_innov)


def armodel_residual(params, inputs, stateini=0, fillnan=False):
    ''' Compute residuals of an AR model.
    If there are nan in inputs, the function produces nan, but
    the internal states are kept in memory.

    Parameters
    -----------
    params : float or np.ndarray
        AR coefficient. If a float is given, the
        value is repeated n times across the time series
    inputs : numpy.ndarray
        AR1 time series. [n, p] array
        - n is the number of time steps
        - p is the number of time series to process
    stateini : float
        Initial condition
    fillnan : bool
        Infill nan values with stateini or not.

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

    '''
    if not HAS_C_STAT_MODULE:
        raise ValueError('C module c_hydrodiy_stat is not available, '+\
                'please run python setup.py build')

    stateini = np.float64(stateini)

    shape_inputs = inputs.shape
    inputs = np.atleast_2d(inputs).astype(np.float64)
    fillnan = np.int32(fillnan)

    # Transpose 1d array
    if inputs.shape[0] == 1:
        inputs = inputs.T

    shape = inputs.shape

    # set the array contiguous to work with C
    if not inputs.flags['C_CONTIGUOUS']:
        inputs = np.ascontiguousarray(inputs)

    # Set params
    params = np.atleast_1d(params).astype(np.float64)

    # Initialise innov
    residuals = np.zeros(shape, np.float64)

    # Run model
    ierr = c_hydrodiy_stat.armodel_residual(stateini, fillnan, \
                        params, inputs, residuals)
    if ierr!=0:
        raise ValueError('c_hydrodiy_stat.armodel_residual returns %d'%ierr)

    return np.reshape(residuals, shape_inputs)

