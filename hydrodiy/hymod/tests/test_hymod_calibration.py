import os
import re
import unittest

from timeit import Timer
import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hymod.model import Model
from hymod.calibration import Calibration


class Dummy(Model):

    def __init__(self):
        Model.__init__(self, 'dummy',
            nconfig=1,\
            ninputs=2, \
            nparams=2, \
            nstates=2, \
            noutputs_max = 2,
            inputs_names = ['I1', 'I2'], \
            outputs_names = ['O1', 'O2'])

        self.config.names = 'debug'
        self.states.names = ['State1', 'State2']
        self.params.names = ['Param1', 'Param2']
        self.params.units = ['mm', 'mm']
        self.params.min = [0, 0]
        self.params.max = [20, 20]

    def run(self):
        par1 = self.params['Param1']
        par2 = self.params['Param2']

        outputs = par1 * np.cumsum(self.inputs.data, 0) + par2

        self._states.data = outputs[-1, :]

        nvar = self._outputs.nvar
        self._outputs.data = outputs[:, :nvar]


class CalibrationDummy(Calibration):

    def __init__(self):

        model = Dummy()

        Calibration.__init__(self, 
            model = model, \
            ncalparams = 2, \
            timeit = True)

        self.calparams_means.data =  [0, 0]
        self.calparams_stdevs.data = [1, 0, 0, 1]

    def cal2true(self, calparams):
        return np.exp(calparams)

 

class CalibrationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibrationTestCase')

    def test_calibration1(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.inputs.data = inputs

        dum.params.data = params
        dum.run()
        observations = dum.outputs.data[:, 0].copy()

        calib = CalibrationDummy()
        calib.setup(observations, inputs)

        str = '{0}'.format(calib)


    def test_calibration2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        obs = np.random.uniform(0, 1, 1000)
        calib = CalibrationDummy()
       
        try:
            calib.idx_cal = obs==obs
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('No observations data'))


    def test_calibration3(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        obs = np.random.uniform(0, 1, 1000)
        calib = CalibrationDummy()
        calib.setup(obs, inputs)
       
        try:
            start, explo, explo_ofun = calib.explore(iprint=0, nsamples=10)
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('No idx_cal data'))


    def test_calibration4(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 2)
        dum.inputs.data = inputs

        dum.params.data = params
        dum.run()
        obs = dum.outputs.data[:, 0].copy()

        calib = CalibrationDummy()
        calib.setup(obs, inputs)
        calib.idx_cal = obs == obs
       
        start, explo, explo_ofun = calib.explore(iprint=0, nsamples=10)
        final, out, _ = calib.fit(start, iprint=0, ftol=1e-8)

        self.assertTrue(np.allclose(calib.model.params.data, params))


if __name__ == "__main__":
    unittest.main()
