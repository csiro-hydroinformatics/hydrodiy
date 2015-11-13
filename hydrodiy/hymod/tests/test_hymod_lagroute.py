import os
import re
import unittest
import itertools

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from hymod.models.lagroute import LagRoute, CalibrationLagRoute
from hymod import calibration


import c_hymod_models_utils
UHEPS = c_hymod_models_utils.uh_getuheps()


class LagRouteTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> LagRouteTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT


    def test_print(self):
        lr = LagRoute()
        str_lr = '%s' % lr


    def test_run(self):

        ierr_id = ''
        lr = LagRoute()
        lr.allocate(20, 2)
        lr.initialise()
        inputs = np.zeros((20, 1))
        inputs[1,0] = 100
        lr.inputs.data = inputs
        lr.run()


    def test_error1(self):
        lr = LagRoute()
        try:
            lr.allocate(20, 30)
        except ValueError as  e:
            pass

        self.assertTrue(e.message.startswith('Too many outputs defined'))


    def test_error2(self):
        lr = LagRoute()
        lr.allocate(20, 2)
        lr.initialise()

        try:
            lr.inputs.data = np.random.uniform(size=(20, 3))
        except ValueError as  e:
            pass

        self.assertTrue(e.message.startswith('inputs matrix: tried setting _data'))


    def test_sample(self):
        calib = CalibrationLagRoute()
        nsamples = 100
        samples = calib.sample(nsamples)
        self.assertTrue(samples.shape == (nsamples, 2))


    def test_uh1(self):
        lr = LagRoute()
        for u, a in itertools.product(np.linspace(0, 10, 20), \
                np.linspace(0, 1, 20)):
            lr.params.data = [u, a]
            ck = abs(np.sum(lr.uh.data)-1) < UHEPS
            self.assertTrue(ck)


    def test_uh2(self):
        lr = LagRoute()

        dt = 86400 # daily model
        L = 86400 # 86.4 km reach
        qstar = 1 # qstar = 1 m3/s
        theta2 = 1 # linear model
        lr.config.data = [dt, L, qstar, theta2]

        # Set uh
        alpha = 1.
        for U in np.linspace(0.1, 20, 100):

            lr.params.data = [U, alpha]

            ck = abs(np.sum(lr.uh.data)-1) < 1e-5
            self.assertTrue(ck)

            tau = alpha * L * U
            k = int(tau/dt)
            w = tau/dt - k
            ck = abs(lr.uh.data[k]-1+w) < 1e-5
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

        # Set outputs
        lr.allocate(len(inputs), 4)
        lr.inputs.data = inputs

        for theta2 in [1, 2]:

            lr.config.data = [dt, L, qstar, theta2]

            # Run
            UU = np.linspace(0.1, 20, 20)
            aa = np.linspace(0., 1., 20)
            dta = 0
            count = 0

            for U, alpha in itertools.product(UU, aa):

                t0 = time.time()

                lr.params.dat = [U, alpha]
                lr.initialise()
                lr.run()

                t1 = time.time()
                dta += 1000 * (t1-t0) / nval * 365.25

                v0 = 0
                vr = lr.outputs.data[-1, 2]
                v1 = lr.outputs.data[-1, 3]
                si = np.sum(inputs) * dt
                so = np.sum(lr.outputs.data[:,0]) * dt

                B = si - so - v1 - vr + v0
                ck = abs(B/so) < 1e-10

                if not ck:
                    import pdb; pdb.set_trace()

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

        lr.config.data = [dt, L, qstar, theta2]

        # Set outputs
        lr.allocate(len(inputs))
        lr.inputs.data = inputs

        # Run
        for U in range(1, 11):
            lr.params.data = [U, 1.]
            lr.initialise()
            lr.run()

            err = np.abs(lr.outputs.data[U:,0] - inputs[:-U, 0])

            ck = np.max(err) < 1e-10
            self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
