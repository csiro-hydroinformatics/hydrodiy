import math
from pathlib import Path

import pytest
import numpy as np
from scipy import linalg

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvt

from hydrodiy.stat import bayesutils
from hydrodiy.plot import bayesplot, putils

np.random.seed(0)

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"

NCHAINS = 3
NPARAMS = 4
NSAMPLES = 100

# Generate mean vector
MU = np.linspace(0, 1, NPARAMS)

# GenerateCOVariance matrix
rnd = np.random.uniform(-1, 1, (NPARAMS, NPARAMS))
rnd = rnd+rnd.T
eig, vects = linalg.eig(rnd)
COV = np.dot(vects, np.dot(np.diag(np.abs(eig)), vects.T))

# Generate samples
SAMPLES = np.zeros((NCHAINS, NPARAMS, NSAMPLES))
for chain in range(NCHAINS):
    SAMPLES[chain, :, :] = np.random.multivariate_normal(\
                                    mean=MU, cov=COV, size=NSAMPLES).T

# MVT log posterior for the first chain
def logpost(theta):
    """ Mvt logpost """
    mu = theta[:NPARAMS]
    cov, _, _ = bayesutils.vect2cov(theta[NPARAMS:])
    loglike = mvt.logpdf(SAMPLES[0, :, :].T, mean=mu, cov=cov)
    # Jeffreys" prior
    logprior = -(mu.shape[0]+1)/2*math.log(linalg.det(cov))
    return np.sum(loglike)+logprior



def test_slice2d():
    fig, ax = plt.subplots()
    vect, _, _ = bayesutils.cov2vect(COV)
    params = np.concatenate([MU, vect])
    zz, yy, zz = bayesplot.slice2d(ax, logpost, params, \
                                0, 1, 0.5, 0.5)
    fp = FIMG / "bayesplot_slice_2d.png"
    fig.savefig(fp)


def test_slice2d_errors():
    fig, ax = plt.subplots()
    vect, _, _ = bayesutils.cov2vect(COV)
    params = np.concatenate([MU, vect])

    msg = "Expected parameter indexes"
    with pytest.raises(ValueError, match=msg):
        bayesplot.slice2d(ax, logpost, params, \
                                len(params), 1, 0.5, 0.5)


    msg = "Expected dval1"
    with pytest.raises(ValueError, match=msg):
        bayesplot.slice2d(ax, logpost, params, \
                                0, 1, -0.5, 0.5)

    msg = "Expected scale"
    with pytest.raises(ValueError, match=msg):
        bayesplot.slice2d(ax, logpost, params, \
                                0, 1, 0.5, 0.5, scale1="ll")

    msg = "Expected dlogpostmin"
    with pytest.raises(ValueError, match=msg):
        bayesplot.slice2d(ax, logpost, params, \
                                0, 1, 0.5, 0.5, dlogpostmin=-1)

def test_slice2d_log():
    fig, ax = plt.subplots()
    vect, _, _ = bayesutils.cov2vect(COV)
    params = np.concatenate([MU+0.01, vect])
    bayesplot.slice2d(ax, logpost, params, \
                                0, 1, 0.5, 0.5, \
                                scale1="log", scale2="log", \
                                dlogpostmin=10, dlogpostmax=1e-2, nlevels=5)
    fp = FIMG / "bayesplot_slice_2d_log.png"
    fig.savefig(fp)


def test_plotchains():
    fig = plt.figure()
    accept = np.ones(SAMPLES.shape[0])
    bayesplot.plotchains(fig, SAMPLES, accept)
    fp = FIMG / "bayesplot_plotchains.png"
    fig.set_size_inches((18, 10))
    fig.tight_layout()
    fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
