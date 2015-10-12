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

    def test_memusage(self):
        mem = mutils.getmemusage()

    def test_sinmodel(self):

        soy = np.arange(1, 367*86400, 86400)

        mu = 100

        eta = 3

        phase = 0.2
        phi = 2*math.pi * max(0., 
            min(1, math.exp(phase)/(1+math.exp(phase))))

        alpha = 1

        params = [mu, eta, phase, alpha]

        Q1 = mutils.sinmodel(params, soy)

        u = np.sin((0. + soy)/365.2425/86400*2*math.pi + math.pi/2 - phi)

        Q2 = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))
        Q2 = mu + math.exp(eta) * Q2
        self.assertTrue(np.allclose(Q1, Q2))

        alpha = -10.
        params = [mu, eta, phase, alpha]

        Q1 = mutils.sinmodel(params, soy)
        Q2 = (np.exp(alpha*u)-math.exp(-alpha))/(math.exp(alpha)-math.exp(-alpha))
        Q2 = mu + math.exp(eta) * Q2
        self.assertTrue(np.allclose(Q1, Q2))



if __name__ == "__main__":
    unittest.main()
