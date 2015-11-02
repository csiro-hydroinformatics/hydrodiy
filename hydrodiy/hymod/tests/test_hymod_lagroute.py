import os
import re
import unittest
import itertools

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from hymod.model import ModelError
from hymod.models.lagroute import LagRoute
from hymod import errfun


import c_hymod_models_utils
UHEPS = c_hymod_models_utils.uh_getuheps()


class LagRouteTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> LagRouteTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT


    def test_print(self):
        gr = LagRoute()
        str_gr = '%s' % gr


    def test_error1(self):

        ierr_id = ''
        gr = LagRoute()
        gr.create_outputs(20, 30)
        gr.initialise()
        inputs = np.random.uniform(size=(20, 2))

        try:
            gr.run(inputs)
        except ModelError as  e:
            ierr_id = e.ierr_id

        self.assertTrue(ierr_id == 'ESIZE_OUTPUTS')


    def test_get_calparams_sample(self):

        nsamples = 100
        gr = LagRoute()
        samples = gr.get_calparams_samples(nsamples)
        self.assertTrue(samples.shape == (nsamples, 2))


    def test_uh1(self):

        gr = LagRoute()

        for u, a in itertools.product(np.linspace(0, 10, 20), \
                np.linspace(0, 1, 20)):
            gr.set_trueparams([u, a])

            ck = abs(np.sum(gr.uh)-1) < UHEPS
            self.assertTrue(ck)


    def test_uh2(self):

        lr = LagRoute()

        # Set configuration
        dt = 86400 # daily model
        L = 86400 # 86.4 km reach
        qstar = 1 # qstar = 1 m3/s
        theta2 = 1 # linear model
        lr.set_config([dt, L, qstar, theta2])

        # Set uh
        alpha = 1.
        for U in np.linspace(0.1, 20, 100):

            lr.set_trueparams([U, alpha])

            ck = abs(np.sum(lr.uh)-1) < 1e-5
            self.assertTrue(ck)

            tau = alpha * L * U
            k = int(tau/dt)
            w = tau/dt - k
 
            ck = abs(lr.uh[k]-1+w) < 1e-5
                
            self.assertTrue(ck)
 

    def test_massbalance(self):

        nval = 1000
        q1 = np.exp(np.random.normal(0, 2, size=nval))
        inputs = np.ascontiguousarray(q1[:,None])

        lr = LagRoute()

        # Set configuration
        dt = 86400 # daily model
        L = 86400 # 86.4 km reach
        qstar = 50 # qstar = 50 m3/s

        for theta2 in [1, 2]:

            lr.set_config([dt, L, qstar, theta2])

            # Set outputs
            lr.create_outputs(len(inputs))

            # Run
            UU = np.linspace(0.1, 20, 20)
            aa = np.linspace(0., 1., 20)
            dta = 0

            for U, alpha in itertools.product(UU, aa):

                t0 = time.time()

                lr.set_trueparams([U, alpha])
                lr.initialise()
                lr.run(inputs)

                t1 = time.time()
                dta += 1000 * (t1-t0) / nval * 365.25

                v0 = 0
                vr = lr.outputs[-1, 2]
                v1 = lr.outputs[-1, 3]
                si = np.sum(inputs) * dt
                so = np.sum(lr.outputs[:,0]) * dt

                B = si - so - v1 - vr + v0
                ck = abs(B/so) < 1e-10

                self.assertTrue(ck)

            dta /= (len(UU) * len(aa))
            print('\t\ttheta2={0} - Average runtime = {1:.5f} ms/yr'.format( \
                theta2, dta))

    def test_lagroute_lag(self):

        nval = 1000
        q1 = np.exp(np.random.normal(0, 2, size=nval))
        inputs = np.ascontiguousarray(q1[:,None])

        lr = LagRoute()

        # Set configuration
        dt = 86400 # daily model
        L = 86400 # 86.4 km reach
        qstar = 50 # qstar = 50 m3/s
        theta2 = 1

        lr.set_config([dt, L, qstar, theta2])

        # Set outputs
        lr.create_outputs(len(inputs))

        # Run
        for U in range(1, 11):
            lr.set_trueparams([U, 1.])
            lr.initialise()
            lr.run(inputs)

            err = np.abs(lr.outputs[U:,0] - inputs[:-U, 0])

            ck = np.max(err) < 1e-10
            self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
