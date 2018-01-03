import os, math

import unittest
import numpy as np

from hydrodiy.stat import bayesutils

np.random.seed(0)

class MCMCStatTestCase(unittest.TestCase):


    def setUp(self):
        print('\t=> MCMCStatTestCase (hystat)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        self.nchains = 5
        self.nparams = 10
        self.nsamples = 5000
        self.samples = np.random.uniform(0, 1,\
                        (self.nchains, self.nparams, self.nsamples))


    def test_is_semidefinitepos(self):
        ''' Test semi-definite positive '''

        nvar = 5

        # Build semi-positive definite matrix
        A = np.random.uniform(-1, 1, (nvar, nvar))
        A = A+A.T
        eig, vects = np.linalg.eig(A)
        eig = np.abs(eig)
        A = np.dot(vects, np.dot(np.diag(eig), vects.T))
        isok = bayesutils.is_semidefinitepos(A)
        self.assertTrue(isok is True)

        # Build a non semi-definite positive matrix
        # from eigen values and unit matrix
        eig[0] = -1
        A = np.dot(vects, np.dot(np.diag(eig), vects.T))
        isok = bayesutils.is_semidefinitepos(A)
        self.assertTrue(isok is False)


    def test_ldl_decomp(self):
        ''' Test LDL decomposition '''

        nvar = 5

        # Build semi-definite positive matrix
        A = np.random.uniform(-1, 1, (nvar, nvar))
        A = A+A.T
        eig, vects = np.linalg.eig(A)
        eig = np.abs(eig)
        A = np.dot(vects, np.dot(np.diag(eig), vects.T))

        # Compute LDL decomposition
        L, D = bayesutils.ldl_decomp(A)

        self.assertTrue(np.allclose(np.diag(L), 1.))
        self.assertTrue(L.shape == (nvar, nvar))
        self.assertTrue(D.shape == (nvar, ))

        A2 = np.dot(L, np.dot(np.diag(D), L.T))
        self.assertTrue(np.allclose(A, A2))


    def test_cov2sigscorr(self):
        ''' Test the computation of sigs and corr from  cov '''

        nvar = 10

        # Build semi-definite positive matrix
        A = np.random.uniform(-1, 1, (nvar, nvar))
        A = A+A.T
        eig, vects = np.linalg.eig(A)
        eig = np.abs(eig)
        cov = np.dot(vects, np.dot(np.diag(eig), vects.T))

        sigs, corr = bayesutils.cov2sigscorr(cov)
        self.assertTrue(len(sigs) == nvar)
        self.assertTrue(np.all(sigs>0))

        offdiag = corr[np.triu_indices(nvar, 1)]
        self.assertTrue(np.all((offdiag>-1) & (offdiag<1)))
        self.assertTrue(np.allclose(np.diag(corr), 1))


    def test_mucov2vect(self):
        ''' Test transformation from  parameter vector to mu/cov '''
        nvars = 5
        nval = nvars+nvars*(nvars-1)//2
        vect = np.random.uniform(-2, 2, nval)

        cov, sigs2, coefs = bayesutils.vect2cov(vect)

        vect2, sigs2b, coefsb = bayesutils.cov2vect(cov)

        self.assertTrue(np.allclose(vect, vect2))

        covc, sigs2c, coefsc = bayesutils.vect2cov(vect2)
        self.assertTrue(np.allclose(cov, covc))
        self.assertTrue(np.allclose(sigs2b, sigs2c))
        self.assertTrue(np.allclose(coefs, coefsc))


    def test_gelman(self):
        ''' Test Gelman convergence stat '''
        Rc = bayesutils.gelman(self.samples)
        self.assertTrue(Rc.shape == (self.nparams, ))
        self.assertTrue(np.all((Rc>1) & (Rc<1.001)))


    def test_laggedcorr(self):
        ''' Test autocorrelation stat '''
        lagc = bayesutils.laggedcorr(self.samples)
        self.assertTrue(lagc.shape == (self.nchains, self.nparams, 10))
        self.assertTrue(np.all((lagc>-0.1) & (lagc<0.1)))



if __name__ == "__main__":
    unittest.main()
