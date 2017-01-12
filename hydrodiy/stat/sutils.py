import re
from math import sqrt
import numpy as np
import pandas as pd
import c_hydrodiy_stat


def ppos(nval, cst=0.3):
    ''' Compute plotting position for sample of size nval

    Parameters
    -----------
    nval : int
        Sample size
    cst : float
        Plotting position constant

    Returns
    -----------
    ppos : numpy.ndarray
        Plotting postion
    '''
    return (np.arange(1, nval+1)-cst)/(nval+1-2*cst)


def ar1innov(params, innov):
    ''' Compute AR1 time series from innovation

    Parameters
    -----------
    params : list
        Parameter vector with. [2] array:
        - params[0] ar1 coefficient
        - params[1] Initial value of the innovation
    innov : numpy.ndarray
        Innovation time series. [n, p] array:
        - n is the number of time steps
        - p is the number of time series to process

    Returns
    -----------
    outputs : numpy.ndarray
        Time series of AR1 simulations. [n, p] or [n] array
        if p=1

    Example
    -----------
    >>> import numpy as np
    >>> from hystat import sutils
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov([0.95, 0.], innov1)
    >>> innov2 = sutils.ar1inverse([0.95, 0.], data)
    >>> np.allclose(innov1, innov2)
    True

    '''
    shape = innov.shape
    innov = np.atleast_2d(innov).astype(np.float64)

    # Transpose 1d array
    if innov.shape[0] == 1:
        innov = innov.T

    # set the array contiguous to work with C
    if not innov.flags['C_CONTIGUOUS']:
        innov = np.ascontiguousarray(innov)

    outputs = np.zeros(innov.shape, np.float64)

    params = np.atleast_1d(params).astype(np.float64)
    ierr = c_hydrodiy_stat.ar1innov(params, innov, outputs)
    if ierr!=0:
        raise ValueError('ar1innov returns %d'%ierr)

    return np.reshape(outputs, shape)


def ar1inverse(params, inputs):
    ''' Compute innovations from an AR1 time series

    Parameters
    -----------
    params : list
        Parameter vector with. [2] array:
        - params[0] ar1 coefficient
        - params[1] Initial value of the innovation
    inputs : numpy.ndarray
        AR1 time series. [n, p] array
        - n is the number of time steps
        - p is the number of time series to process

    Returns
    -----------
    innov : numpy.ndarray
        Time series of innovations. [n, p] array

    Example
    -----------
    >>> import numpy as np
    >>> from hydrodiy.stat import sutils
    >>> nval = 100
    >>> innov1 = np.random.normal(size=nval)
    >>> data = sutils.ar1innov([0.95, 0.], innov1)
    >>> innov2 = sutils.ar1inverse([0.95, 0.], data)
    >>> np.allclose(innov1, innov2)
    True

    '''
    shape = inputs.shape
    inputs = np.atleast_2d(inputs).astype(np.float64)

    # Transpose 1d array
    if inputs.shape[0] == 1:
        inputs = inputs.T

    # set the array contiguous to work with C
    if not inputs.flags['C_CONTIGUOUS']:
        inputs = np.ascontiguousarray(inputs)

    innov = np.zeros(inputs.shape, np.float64)

    params = np.atleast_1d(params).astype(np.float64)
    ierr = c_hydrodiy_stat.ar1inverse(params, inputs, innov)
    if ierr!=0:
        raise ValueError('c_hystat.ar1inverse returns %d'%ierr)

    return np.reshape(innov, shape)


def lhs(nparams, nsample, pmin, pmax):
    ''' Latin hypercube sampling

    Parameters
    -----------
    nparams : int
        Number of parameters
    nsample : int
        Number of sample to draw
    pmin : list
        Lower bounds of parameters
    pmax : list
        Upper bounds of parameters

    Returns
    -----------
    samples : numpy.ndarray
        Parameter samples (nsamples x nparams)

    Example
    -----------
    >>> nparams = 5; nsamples = 3
    >>> sutils.lhs(nparams, nsamples)

    '''
    # Process pmin and pmax
    pmin = np.atleast_1d(pmin)
    if pmin.shape[0] == 1:
        pmin = np.repeat(pmin, nparams)

    if len(pmin) != nparams:
        raise ValueError(('Expected pmin of length' + \
            ' {0}, got {1}').format(nparams, \
                len(pmin)))

    pmax = np.atleast_1d(pmax)
    if pmax.shape[0] == 1:
        pmax = np.repeat(pmax, nparams)

    if len(pmax) != nparams:
        raise ValueError(('Expected pmax of length' + \
            ' {0}, got {1}').format(nparams, \
                len(pmax)))

    # Initialise
    samples = np.zeros((nsample, nparams))

    # Sample
    for i in range(nparams):
        du = float(pmax[i]-pmin[i])/nsample
        u = np.linspace(pmin[i]+du/2, pmax[i]-du/2, nsample)

        kk = np.random.permutation(nsample)
        s = u[kk] + np.random.uniform(-du/2, du/2, size=nsample)

        samples[:, i] = s

    return samples

