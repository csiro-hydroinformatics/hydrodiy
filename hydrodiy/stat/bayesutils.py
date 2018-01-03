import math
import pandas as pd
import numpy as np

from itertools import product as prod


def is_semidefinitepos(matrix):
    ''' Check if matrix is semi-definite positive

    Parameters
    -----------
    matrix : numpy.ndarray
        Input matrix

    Returns
    -----------
    isok : bool
        True if the matrix is semi-definite positive
        False if not.
    '''
    try:
        chol = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError as err:
        return False


def ldl_decomp(matrix):
    ''' Performs the LDL Cholesky decomposition

    Parameters
    -----------
    matrix : numpy.ndarray
        Input matrix

    Returns
    -----------
    Lmat : numpy.ndarray
        L matrix (lower triangular with one on diagonal)
    Dmat : numpy.ndarray
        D vector (positive numbers)
    '''

    # Apply Cholesky decomposition
    try:
        Tmat = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as err:
        raise ValueError('Expected matrix to be semi-definite positive')

    # extract the D matrix
    diag = np.diag(Tmat)
    Dmat = diag*diag

    # extract L matrix
    Lmat = np.dot(Tmat, np.diag(1./diag))

    return Lmat, Dmat


def cov2sigscorr(cov):
    ''' Extract correlation and standard deviation from  covariance
    matrix

     Parameters
    -----------
    cov : numpy.ndarray
        Covariance matrix

    Returns
    -----------
    sigs : numpy.ndarray
        Standard deviations
    corr : numpy.ndarray
        Correlation matrix
    '''
    diag = np.diag(cov)
    sigs = np.sqrt(np.diag(cov))
    fact = np.diag(1./sigs)
    corr = np.dot(fact, np.dot(cov, fact))

    return sigs, corr


def cov2vect(cov):
    ''' Convert a covariance matrix to a parameter vector.

    Covariance matrix is inverted to get the
    precision matrix, which is further decomposed
    using the LDL Cholesky decomposition. The D vector is finally
    converted to log and the of-diagonal element of L are extracted.

    Parameters
    -----------
    cov : numpy.ndarray
        Covariance matrix

    Returns
    -----------
    vect : numpy.ndarray
        1D parameter vector
    sigs2 : numpy.ndarray
        Squared standard deviation of random errors
    coefs : numpy.ndarray
        Regression coefficients
    '''

    # Check covariance matrix size
    nvars, _ = cov.shape

    # Initialise
    vect = np.zeros(nvars+nvars*(nvars-1)//2)

    # LDL decomposition of precision matrix
    precis = np.linalg.inv(cov)
    coefs, sigs2 = ldl_decomp(precis)

    # Store standard deviation squared
    vect[:nvars] = np.log(sigs2)

    # Store regression coefficients
    vect[nvars:] = coefs[np.tril_indices(nvars, -1)]

    return vect, sigs2, coefs


def vect2cov(vect):
    ''' Convert a parameter vector to a covariance matrix.

    The parameter vector stores the elements
    of the LDL Cholesky decomposition of the covariance matrix.
    The D matrix is stored in log transformed form.

    Parameters
    -----------
    vect : numpy.ndarray
        1D parameter vector

    Returns
    -----------
    cov : numpy.ndarray
        Covariance matrix
    sigs2 : numpy.ndarray
        Squared standard deviation of random errors
    coefs : numpy.ndarray
        Regression coefficients
    '''
    nval = len(vect)

    # Compute the number of variables
    nvars = (math.sqrt(1+8*nval)-1)/2

    if abs(nvars-round(nvars))>1e-8:
        raise ValueError('Expected integer solution for '+\
                'the number of parameters, got {0}'.format(nvars))
    else:
        nvars = int(round(nvars))

    # Extract sigs
    sigs2 = np.exp(vect[:nvars])

    # Extract regression coefficients
    coefs = np.eye(nvars)
    coefs[np.tril_indices(nvars, -1)] = vect[nvars:]

    # Build precision matrix
    precis = np.dot(coefs, np.dot(np.diag(sigs2), coefs.T))

    # Get covariance matrix
    cov = np.linalg.inv(precis)

    return cov, sigs2, coefs


def gelman_convergence(samples):
    ''' Compute the convergence statistic advocated by Gelman

    Parameters
    -----------
    samples : numpy.ndarray
        3D array containing the MCMC samples. The dimension of
        the array are:
        * dim 0 : Number of chains
        * dim 1 : Number of parameters
        * dim 2 : Number of sample for each chain and parameters

    Returns
    -----------
    Rc : numpy.ndarray
        Rc statistic for each parameter.
        A value between 1.000 and 1.002 is expected.
    '''

    # Get sample dimensions
    nchains, nparams, nsamples = samples.shape

    # Compute mean and var of each chain
    stats = np.concatenate([\
                np.mean(samples, axis=2)[:, :, None], \
                np.var(samples, axis=2)[:, :, None], \
            ], axis=2)

    # Overall mean
    means = np.mean(stats[:, :, 0], axis=0)

    # Compute Gelman stat
    between = nsamples/(nchains-1) \
                    * np.sum((stats[:, :, 0]-means[None, :])**2, axis=0)

    within = np.mean(stats[:, :, 1], axis=0)

    var_est = nsamples/(nsamples-1)*within\
                +(nchains+1)/(nchains*nsamples)*between

    degfree = nsamples-nparams
    gelmanR = np.sqrt((degfree+3)/(degfree+1)*var_est/within)

    return gelmanR


def laggedcorr(samples, maxlag=10):
    ''' Lagged correlation of MCMC samples

    Parameters
    -----------
    samples : numpy.ndarray
        3D array containing the MCMC samples. The dimension of
        the array are:
        * dim 0 : Number of chains
        * dim 1 : Number of parameters
        * dim 2 : Number of sample for each chain and parameters
    maxlag : int
        Maximum number of lags

    Returns
    -----------
    lagc : numpy.ndarray
        Rc statistic for each parameter
    '''

    # Get sample dimensions
    nchains, nparams, nsamples = samples.shape

    # initialise
    lagc = np.zeros((nchains, nparams, maxlag))

    # Compute lagged correlations
    for i, j, lag in prod(range(nchains), range(nparams), \
                                            range(1, maxlag+1)):
        lagc[i, j, lag-1] = np.corrcoef(\
                    samples[i, j, :-lag], samples[i, j, lag:])[0, 1]

    return  lagc


