import math
from pathlib import Path
import pytest

import unittest
import numpy as np
from scipy import linalg

from hydrodiy.stat import bayesutils

np.random.seed(0)

FTEST = Path(__file__).resolve().parent

NCHAINS = 5
NPARAMS = 10
NSAMPLES = 5000
SAMPLES = np.random.uniform(0, 1,\
           (NCHAINS, NPARAMS, NSAMPLES))


def test_is_semidefinitepos():
    """ Test semi-definite positive """

    nvar = 5

    # Build semi-positive definite matrix
    A = np.random.uniform(-1, 1, (nvar, nvar))
    A = A+A.T
    eig, vects = linalg.eig(A)
    eig = np.abs(eig)
    A = np.dot(vects, np.dot(np.diag(eig), vects.T))
    isok = bayesutils.is_semidefinitepos(A)
    assert (isok is True)

    # Build a non semi-definite positive matrix
    # from eigen values and unit matrix
    eig[0] = -1
    A = np.dot(vects, np.dot(np.diag(eig), vects.T))
    isok = bayesutils.is_semidefinitepos(A)
    assert (isok is False)


def test_ldl_decomp():
    """ Test LDL decomposition """

    nvar = 5

    # Build semi-definite positive matrix
    A = np.random.uniform(-1, 1, (nvar, nvar))
    A = A+A.T
    eig, vects = linalg.eig(A)
    eig = np.abs(eig)
    A = np.dot(vects, np.dot(np.diag(eig), vects.T))

    # Compute LDL decomposition
    L, D = bayesutils.ldl_decomp(A)

    assert (np.allclose(np.diag(L), 1.))
    assert (L.shape == (nvar, nvar))
    assert (D.shape == (nvar, ))

    A2 = np.dot(L, np.dot(np.diag(D), L.T))
    assert (np.allclose(A, A2))


def test_cov2sigscorr():
    """ Test the computation of sigs and corr from  cov """

    nvar = 10

    # Build semi-definite positive matrix
    A = np.random.uniform(-1, 1, (nvar, nvar))
    A = A+A.T
    eig, vects = linalg.eig(A)
    eig = np.abs(eig)
    cov = np.dot(vects, np.dot(np.diag(eig), vects.T))

    sigs, corr = bayesutils.cov2sigscorr(cov)
    assert (len(sigs) == nvar)
    assert (np.all(sigs>0))

    offdiag = corr[np.triu_indices(nvar, 1)]
    assert (np.all((offdiag>-1) & (offdiag<1)))
    assert (np.allclose(np.diag(corr), 1))


def test_mucov2vect():
    """ Test transformation from  parameter vector to mu/cov """
    nvars = 5
    nval = nvars+nvars*(nvars-1)//2
    vect = np.random.uniform(-2, 2, nval)

    cov, sigs2, coefs = bayesutils.vect2cov(vect)

    vect2, sigs2b, coefsb = bayesutils.cov2vect(cov)

    assert (np.allclose(vect, vect2))

    covc, sigs2c, coefsc = bayesutils.vect2cov(vect2)
    assert (np.allclose(cov, covc))
    assert (np.allclose(sigs2b, sigs2c))
    assert (np.allclose(coefs, coefsc))


def test_gelman_convergence():
    """ Test Gelman convergence stat """
    Rc = bayesutils.gelman_convergence(SAMPLES)
    assert (Rc.shape == (NPARAMS, ))
    assert (np.all((Rc>1) & (Rc<1.001)))


def test_laggedcorr():
    """ Test autocorrelation stat """
    lagc = bayesutils.laggedcorr(SAMPLES)
    assert (lagc.shape == (NCHAINS, NPARAMS, 10))
    assert (np.all((lagc>-0.1) & (lagc<0.1)))

