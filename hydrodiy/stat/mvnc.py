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


def __compute_log_cdf(cdf):
    ''' Function to avoid computing log for
    extremely low values of cdf causing floating point errors
    '''
    # initialise log_cdf to 0
    log_cdf = 0.*cdf

    # compute log(cdf) for strictly positive values only
    idx_cdf_pos = cdf>0.
    if np.sum(idx_cdf_pos)>0:
        log_cdf[idx_cdf_pos] = np.log(cdf[idx_cdf_pos])

    return log_cdf



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
        idx_case = cases == icase
        if np.sum(idx_case)>0:

            if icase == 0:
                # No variables are censorsed
                # standard multivariate normal pdf
                logpdf_values[idx_case] = mvt.logpdf(x[idx_case], mean=mu, cov=cov)

            elif icase == 2**nvar-1:
                # All variables are censorsed

                # Get parameters
                sig, correl = __get_sig_corr(cov)

                if nvar > 1:
                    # all censorsed
                    lower = np.zeros(nvar) # Does not matter here
                    upper = (censors-mu)/sig * np.ones(nvar)
                    infin = np.zeros(nvar)
                        # <0=[-inf, +inf]
                        # 0=[-inf, upper]
                        # 1=[lower, +inf]
                        # 2=[lower, upper]

                    err, cdf, info = mvn.mvndst(lower, upper, infin, correl)
                else:
                    cdf = norm.cdf(censors, loc=mu, scale=sig)

                log_cdf = 0.
                if cdf > 0:
                    log_cdf = math.log(cdf)
                logpdf_values[idx_case] = log_cdf

            else:
                # Some variables censored, but not all

                # Parameter for non-censorsed
                mu1 = mu[~censvars]
                cov11 = cov[~censvars][:, ~censvars]

                # Parameter for censorsed
                mu2 = mu[censvars]
                cov22 = cov[censvars][:, censvars]

                # Remaining covariance matrix
                cov12 = cov[~censvars][:, censvars]

                # Conditional parameters to compute p(x_c|x_nc)
                # see
                # https://en.wikipedia.org/w/index.php?title=Multivariate_normal_distribution&section=18#Conditional_distributions

                values = x[idx_case][:, ~censvars]-mu1
                if nvar-ncensored > 1:
                    mu3 = mu2 + np.dot(cov12.T, np.dot(np.linalg.inv(cov11), \
                                                            values.T))
                    mu3 = mu3.T

                    cov3 = cov22 - np.dot(cov12.T, \
                            np.dot(np.linalg.inv(cov11), cov12))
                else:
                    mu3 = mu2 + cov12/cov11*values
                    cov3 = cov22 - cov12**2/cov11

                # More than 2d censoring
                if ncensored>1:
                    # Get parameters
                    sig3, correl3 = __get_sig_corr(cov3)

                    # Compute
                    # int[p(x_c=y|x_nc), y, -inf, censor]
                    lower = np.zeros(ncensored) # Does not matter here
                    upper = (censors[censvars]-mu3)/sig3 * np.ones(ncensored)
                    infin = np.zeros(ncensored)
                        # <0=[-inf, +inf]
                        # 0=[-inf, upper]
                        # 1=[lower, +inf]
                        # 2=[lower, upper]

                    # Compute cdf for each point
                    # Very slow - to be improved
                    cdf = np.zeros(np.sum(idx_case))
                    for k in range(len(upper)):
                        err, cdf[k], info = mvn.mvndst(lower, upper[k], infin, correl3)

                else:
                    # Compute
                    # int[p(x_c=y|x_nc), y, -inf, censor]
                    cdf = norm.cdf(censors[censvars], loc=mu3, scale=cov3)

                # Finally, computing log pdf
                # int[p(x_nc, x_c=y), y, -inf, censor] = p(x_nc) int[p(x_c=y|x_nc), y, -inf, censor]
                logpdf_values[idx_case] = mvt.logpdf(x[idx_case][:, ~censvars], \
                                mean=mu1, cov=cov11)+ __compute_log_cdf(cdf)

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

