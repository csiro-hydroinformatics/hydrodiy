import math
import numpy as np

from scipy.stats import mvn, norm
from scipy.stats import multivariate_normal as mvt
from scipy.linalg import toeplitz

EPS = 1e-10

def __get_sig_corr(cov):
    ''' Extract standard deviation and correlation matrix
    from covariance matrix '''
    nvar = cov.shape[0]

    # Standard deviation of each variables
    sig = np.sqrt(np.diag(cov))

    # Correlations in a format expected by mvt
    # (i.e. only of diagonal elements)
    fact = np.diag(1./sig)
    correl = np.dot(fact, np.dot(cov, fact))
    idxr, idxc = np.triu_indices(nvar, 1)
    correl = correl[idxr, idxc]

    return sig, correl


def __get_censors(censors, nvar):
    ''' Generate proper censors vector '''
    censors = np.atleast_1d(censors)

    if censors.shape[0] == 1:
        censors = np.repeat(censors, nvar)

    if not len(censors.shape) == 1:
        raise ValueError(('Censor is not a vector, ' + \
            'its dimensions are {0}').format(censors.shape))

    if not censors.shape[0] == nvar:
        raise ValueError('Expected length of censors to be {0}, but got {1}'.format(\
            nvar, censors.shape[0]))

    return censors

def toeplitz_cov(sig, rho):
    ''' Generate a toeplitz covariance matrix (analogous to AR1)

    Parameters
    -----------
    sig : np.ndarray
        Standard deviations of variables. [p] array.
    rho : float
        Correlation

    Returns
    -----------
    cov : numpy.ndarray
        Covariance matrix. [p, p] array.
    '''
    # Create a diagonal matrix with standard deviations
    sig = np.diag(sig)
    nvar = sig.shape[0]

    # Generate covariance matrix
    cov = np.dot(sig, np.dot(toeplitz(rho**np.arange(nvar)), sig))

    return cov


def logpdf(x, mu, cov, censors=0):
    ''' Compute the log pdf of a left censored multivariate normal

    Parameters
    -----------
    x : np.ndarray
        Samples. [n, p] array:
        - n is the number of samples
        - p is the number of variables
    mu : np.ndarray
        Location parameters. [p] array.
    cov : np.ndarray
        Covariance matrix. [p, p] array.
    censors : np.ndarray
        Left censoring thresholds. [1] or [p] array.
        If [1] array, the threshold is replicated p times.

    Returns
    -----------
    logpdf_values : numpy.ndarray
        Log likelihood for each sample
    '''

    # Check inputs
    x = np.atleast_2d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    nval, nvar = x.shape

    if mu.shape[0] != nvar:
        raise ValueError('Expected mu of length {0}, but got {1}'.format(\
            nvar, mu.shape[0]))

    if cov.shape != (nvar, nvar):
        raise ValueError('Expected cov of dim {0}, but got {1}'.format(\
            (nvar, nvar), cov.shape))

    censors = __get_censors(censors, nvar)

    # Define the censorsing cases
    # 0 => all < censors
    # 1 => x[0] > censors and rest < censors
    # ...
    cases = np.sum((x-censors < EPS).astype(int) * 2**np.arange(nvar), axis=1)

    # Initialise
    logpdf_values = np.zeros(nval)

    # Loop through cases
    for icase in range(cases.max()+1):

        # Define which variables are censorsed using bitwise operations
        censvars = np.array([bool(icase & 2**n) for n in range(nvar)])
        ncensored = np.sum(censvars)

        # Check case exists in the sample
        idx = cases == icase
        if np.sum(idx)>0:

            if icase == 0:
                # No variables are censorsed
                # standard multivariate normal pdf
                logpdf_values[idx] = mvt.logpdf(x[idx], mean=mu, cov=cov)

            elif icase == 2**nvar-1:
                # All variables are censorsed

                # Get parameters
                sig, correl = __get_sig_corr(cov)

                # all censorsed
                lower = np.zeros(nvar) # Does not matter here
                upper = (censors-mu)/sig * np.ones(nvar)
                infin = np.zeros(nvar)
                    # <0=[-inf, +inf]
                    # 0=[-inf, upper]
                    # 1=[lower, +inf]
                    # 2=[lower, upper]
                err, cdf, info = mvn.mvndst(lower, upper, infin, correl)

                logpdf_values[idx] = math.log(cdf)


            else:
                # Some variables censorsed, but not all of them

                # Parameter for non-censorsed
                mu1 = mu[~censvars]
                cov11 = cov[~censvars][:, ~censvars]

                # Parameter for censorsed
                mu2 = mu[censvars]
                cov22 = cov[censvars][:, censvars]

                # Remaining covariance matrix
                cov12 = cov[~censvars][:, censvars]

                # More than 2d censorsing
                if ncensored>1:
                    # Get parameters
                    sig22, correl22 = __get_sig_corr(cov22)

                    # log pdf of censorsed part
                    lower = np.zeros(ncensored) # Does not matter here
                    upper = (censors[censvars]-mu[censvars])/sig22 * np.ones(ncensored)
                    infin = np.zeros(ncensored)
                        # <0=[-inf, +inf]
                        # 0=[-inf, upper]
                        # 1=[lower, +inf]
                        # 2=[lower, upper]
                    err, cdf, info = mvn.mvndst(lower, upper, infin, correl22)
                else:
                    cdf = norm.cdf(censors[censvars], loc=mu2, scale=cov22)

                # To avoid extremely low values of cdf causing floating point
                # problem
                log_cdf = 0.
                if cdf>0.:
                    log_cdf = math.log(cdf)

                # This is the conditional likelihood p(x_nc|x_c)
                # see
                # https://en.wikipedia.org/w/index.php?title=Multivariate_normal_distribution&section=18#Conditional_distributions
                #
                mu3 = mu1 + np.dot(cov12, \
                        np.dot(np.linalg.inv(cov22), censors[censvars]-mu2))
                cov3 = cov11 - np.dot(cov12, \
                        np.dot(np.linalg.inv(cov22), cov12.T))

                # Finally, computing log pdf
                # p(x_nc, x_c) = p(x_nc|x_c)*p(x_c)
                logpdf_values[idx] = mvt.logpdf(x[idx][:, ~censvars], \
                                mean=mu3, cov=cov3)+log_cdf

    return logpdf_values


def sample(nsamples, mu, cov, censors=0):
    ''' Sample form a left censorsed multivariate normal

    Parameters
    -----------
    nsamples : in
        Number of samples
    mu : np.ndarray
        Location parameters. [p] array.
    cov : np.ndarray
        Covariance matrix. [p, p] array.
    censors : np.ndarray
        Left censoring thresholds. [1] or [p] array.
        If [1] array, the threshold is replicated p times.

    Returns
    -----------
    samples : numpy.ndarray
        Samples. [nsamples, p] array
    '''
    # Check inputs
    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    nvar = mu.shape[0]
    if cov.shape != (nvar, nvar):
        raise ValueError('Expected dimensions of cov to be {0}, but got {1}'.format(\
            (nvar, nvar), cov.shape))

    censors = __get_censors(censors, nvar)

    # Sampling before censoring
    samples = np.random.multivariate_normal(mean=mu, \
                        cov=cov, size=nsamples)

    # Left censoring
    samples = np.maximum(samples, censors)

    return samples

