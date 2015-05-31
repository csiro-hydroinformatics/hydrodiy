import os
import unittest
import numpy as np
import datetime
import math
from hymod import mutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hymod)')
        self.dt = None # TODO

    def test_turc_mezentsev(self):
        P = np.linspace(700, 1200, 20)
        PE = np.linspace(1000, 2000, 20)
        n = 2.3
        Q1 = mutils.turc_mezentsev(n, P, PE)
        Q2 = P*(1-1/(1+(P/PE)**n)**(1/n))
        self.assertTrue(all(Q1==Q2))

    def test_memusage(self):
        mem = mutils.getmemusage()

    def test_sinmodel(self):

        soy = np.arange(1, 367*86400, 86400)

        mu = 100
        eta = 3
        phi = 2*np.pi*0.2
        alpha = 1
        params = [mu, eta, phi, alpha]
        Q1 = mutils.sinmodel(params, soy)

        u = np.sin((0. + soy)/365.2425/86400*2*np.pi + np.pi/2 - phi)
        Q2 = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))
        Q2 = mu + math.exp(eta)/2 * Q2
        self.assertTrue(np.allclose(Q1, Q2))

        alpha = -10.
        params = [mu, eta, phi, alpha]
        Q1 = mutils.sinmodel(params, soy)
        Q2 = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))
        Q2 = mu + math.exp(eta)/2 * Q2
        self.assertTrue(np.allclose(Q1, Q2))



if __name__ == "__main__":
    unittest.main()
