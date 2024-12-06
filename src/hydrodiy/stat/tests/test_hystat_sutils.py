import math
from pathlib import Path
import pytest
import numpy as np
from itertools import product as prod

from scipy.special import kolmogorov
from scipy.linalg import toeplitz
from scipy.stats import norm, anderson

import matplotlib.pyplot as plt

from hydrodiy import has_c_module
from hydrodiy.io import csv
from hydrodiy.stat import sutils, armodels

np.random.seed(5446)
FTEST = Path(__file__).resolve().parent

def test_ppos(allclose):
    nval = 100

    pp = sutils.ppos(nval)
    assert pp[0]>0
    assert pp[-1]<1
    ppd = np.diff(pp)
    assert np.all(ppd>0)

    pp = sutils.ppos(nval, 0.)
    assert allclose(pp, np.arange(1., nval+1.)/(nval+1))


def test_acf_error():
    """ Test acf error """
    data = np.random.uniform(size=20)
    msg = f"Expected idx"
    with pytest.raises(ValueError, match=msg):
        acf, cov = sutils.acf(data, idx=(data>0.5)[2:])


def test_acf_all(allclose):
    """ Test acf """
    if not has_c_module("stat", False):
        pytest.skip("Missing C module c_hydrodiy_stat")

    nval = 100000
    rho = 0.8
    sig = 2
    innov = np.random.normal(size=nval, \
                    scale=sig*math.sqrt(1-rho**2))
    x = armodels.armodel_sim(rho, innov)

    maxlag = 10
    acf, cov = sutils.acf(x, maxlag)

    # Theoretical ACF for AR1 process
    expected = rho**np.arange(1, maxlag+1)
    assert allclose(acf, expected, atol=2e-2)


def test_acf_r(allclose):
    """ Compare acf with R """
    for i in range(1, 5):
        fd = FTEST / "data" / f"acf{i}_data.csv"
        data, _ = csv.read_csv(fd)
        data = np.squeeze(data.values)

        fr = FTEST / "data" / f"acf{i}_result.csv"
        expected, _ = csv.read_csv(fr)
        expected = expected["acf"].values[1:]

        acf, cov = sutils.acf(data, expected.shape[0])

        # Modify for bias
        nval = len(data)
        nacf = len(acf)
        factor = np.arange(nval-nacf, nval)[::-1].astype(float)/nval
        acf *= factor

        assert allclose(expected, acf)


def test_acf_idx(allclose):
    """ Test acf with index selection """
    if not has_c_module("stat", False):
        pytest.skip("Missing C module c_hydrodiy_stat")

    nval = 5000
    sig = 2
    rho1 = 0.7
    innov = np.random.normal(size=nval//2,
                        scale=sig*math.sqrt(1-rho1**2))
    x1 = 10*sig + armodels.armodel_sim(rho1, innov)

    rho2 = 0.1
    innov = np.random.normal(size=nval//2,
                        scale=sig*math.sqrt(1-rho2**2))
    x2 = -10*sig + armodels.armodel_sim(rho2, innov)

    data = np.concatenate([x1, x2])

    acf1, cov = sutils.acf(data, idx=data>=0)
    acf2, cov = sutils.acf(data, idx=data<=0)

    assert allclose([acf1[0], acf2[0]], [rho1, rho2],
                        atol=1e-1)


def test_lhs_error():
    """ Test latin-hypercube sampling errors """
    nparams = 10
    nsamples = 50
    pmin = np.ones(nparams)
    pmax = 10.*np.ones(nparams)

    msg = "Expected pmax"
    with pytest.raises(ValueError, match=msg):
        samples = sutils.lhs(nsamples, pmin, pmax[:2])


def test_lhs():
    """ Test latin-hypercube sampling """
    nparams = 10
    nsamples = 50
    pmin = np.ones(nparams)
    pmax = 10.*np.ones(nparams)

    samples = sutils.lhs(nsamples, pmin, pmax)

    for i in range(nparams):
        u = (np.sort(samples[:,i])-pmin[i])/(pmax[i]-pmin[i])
        ff = sutils.ppos(nsamples)

        # Perform two sided KS test on results
        D = np.max(np.abs(u-ff))
        p = kolmogorov(D*math.sqrt(nsamples))
        assert p>0.95


def test_lhs_norm(allclose):
    """ Test lhs for mvt data """
    nsamples = 500000
    nvars = 5
    mean = np.linspace(1, 2, nvars)
    rho = 0.95
    cov = toeplitz(rho**np.arange(nvars))

    samples = sutils.lhs_norm(nsamples, mean, cov)
    meanS = np.mean(samples, axis=0)
    covS = np.cov(samples.T)

    assert allclose(mean, meanS, atol=0, rtol=1e-4)
    assert allclose(cov, covS, atol=0, rtol=1e-2)


def test_standard_normal(allclose):
    """ Test standard normal computation """
    nsamples = 10
    samples = np.random.uniform(size=nsamples)
    unorm, _ = sutils.standard_normal(samples)

    kk = np.argsort(np.argsort(samples))
    expected = norm.ppf((kk+1)/(nsamples+1))
    assert allclose(unorm, expected)

    samples = np.sort(samples)
    unorm1, _ = sutils.standard_normal(samples)
    unorm2, _ = sutils.standard_normal(samples, sorted=True)
    assert allclose(unorm1, unorm2)


def test_standard_normal_ties(allclose):
    """ Test standard normal computation with ties """
    nsamples = 50
    samples = np.random.uniform(size=nsamples)
    idx = samples > 0.6
    samples[idx] = 0.6
    unorm, rk = sutils.standard_normal(samples)

    n = np.sum(idx)
    r = (2*nsamples-n-1)/2
    assert allclose(rk[idx], r)

    kk = np.argsort(np.argsort(samples)).astype(float)
    kk[idx] = r
    expected = norm.ppf((kk+1)/(nsamples+1))
    assert allclose(unorm, expected)

    # Different method for rank calculation
    unorm, rk = sutils.standard_normal(samples, rank_method="min")
    assert allclose(rk[idx], nsamples-n)

    unorm, rk = sutils.standard_normal(samples, rank_method="max")
    assert allclose(rk[idx], nsamples-1)


def test_semicorr(allclose):
    """ Test semicorr """
    nsamples = 50000
    nvars = 2
    rho_true = 0.55
    mean = np.array([0]*nvars)
    cov = np.array([[1, rho_true], [rho_true, 1]])
    samples = sutils.lhs_norm(nsamples, mean, cov)
    u1, r1 = sutils.standard_normal(samples[:, 0])
    u2, r2 = sutils.standard_normal(samples[:, 1])
    unorm = np.column_stack([u1, u2])

    rho, eta, rho_p, rho_m = sutils.semicorr(unorm)

    assert allclose(eta, 0.311, atol=1e-2)
    assert allclose(rho_p, 0.311, atol=5e-2)
    assert allclose(rho_m, 0.311, atol=5e-2)


def test_pareto_front(allclose):
    """ Test pareto front """
    if not has_c_module("stat", False):
        pytest.skip("Missing C module c_hydrodiy_stat")

    for nval, ncol in prod([20, 200], [2, 5]):
        samples = np.random.normal(size=(nval, ncol))

        isdominated = sutils.pareto_front(samples)

        dominated = np.ones((nval, nval))
        for i, j in prod(range(nval), range(nval)):
            diff = samples[j]-samples[i]
            dominated[i, j] = 1 if np.all(diff>0) else 0

        expected = (dominated.sum(axis=1)>0).astype(int)
        assert allclose(isdominated, expected)

    #import matplotlib.pyplot as plt
    #plt.plot(*samples.T, "o", ms=4, alpha=0.8)
    #plt.plot(*samples[expected==0].T, "ro")
    #plt.show()
    #import pdb; pdb.set_trace()


def test_lstsq(allclose):
    #FIXME add more tests with statsmodel results
    nval, nvar = 100, 3
    for repeat in range(10):
        t_true = np.random.uniform(-0.5, 0.5, nvar)
        X = np.random.uniform(0, 1, (nval, nvar))
        std = 1e-4
        err = np.random.normal(0, std, size=nval)
        y = X.dot(t_true)+err
        res, _, _, _ = sutils.lstsq(X, y)
        assert allclose(res.params, t_true, rtol=0, atol=1e-3)

        std = np.std(X.dot(t_true))*2
        err = np.random.normal(0, std, size=nval)
        y = X.dot(t_true)+err
        res, fstats, fpvalues, _ = sutils.lstsq(X, y)
        # Default zero value constraint,
        # so fstat should be equal to tstat**2
        # and get same pvalue
        assert allclose(fstats, res.tstat**2)
        assert allclose(fpvalues, res.tpvalue)


def test_lstsq_constraints():
    #FIXME add more tests with statsmodel results
    nval, nvar, nrepeat = 100, 3, 500
    fpvalues = np.zeros(nrepeat)
    R = np.zeros((1, nvar))
    R[0, -1] = 1
    r = np.array([5])

    for repeat in range(nrepeat):
        t_true = np.random.uniform(1, 2, nvar)
        t_true[-1] = 5-1e-5
        X = np.random.uniform(0, 1, (nval, nvar))
        std = np.std(X.dot(t_true))/2
        err = np.random.normal(0, std, size=nval)
        y = X.dot(t_true)+err
        res, fstats, fpvalues[repeat], _ = sutils.lstsq(X, y, Rtest=[R], \
                                        rtest=[r])

    # Failing to assume that the last param is different from 5
    assert fpvalues.mean()>0.3


def test_lstsq_wikipedia(allclose):
    # See https://en.wikipedia.org/wiki/Ordinary_least_squares
    height = np.array([1.47, 1.50,	1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, \
                                1.73, 1.75, 1.78, 1.80, 1.83])
    weight = np.array([52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, \
                            61.29, 63.11, 64.47, 66.28, 68.10, 69.92, \
                            72.19, 74.46])
    X = np.column_stack([height, height**2])
    R = np.column_stack([np.zeros(2), np.eye(2)])
    r = np.zeros(2)
    res, fstats, fpvalues, Xi = sutils.lstsq(X, weight, \
                add_intercept=True, Rtest=[R], rtest=[r])

    assert res.shape[0] == 3
    assert allclose(res.params, [128.8128, -143.1620, 61.9603])
    assert allclose(res.stderr, [16.3083, 19.8332, 6.0084])
    assert allclose(res.tstat, [7.8986, -7.2183, 10.3122])
    assert allclose(fstats[0], 5471.24)


