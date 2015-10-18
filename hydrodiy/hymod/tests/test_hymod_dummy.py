import os
import re
import unittest

import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hyio import csv

from hywafari import wdata
from hymod.models.dummy import Dummy


class DummyTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> DummyTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT

    def test_print(self):
        gr = Dummy()
        str_gr = '%s' % gr

    def test_get_calparams_sample(self):
        nsamples = 100
        gr = Dummy()
        samples = gr.get_calparams_samples(nsamples)
        self.assertTrue(samples.shape == (nsamples, 1))

    def test_run(self):
        gr = Dummy()

        nval = 10
        inputs = np.random.uniform(0, 1, size=(nval, 2))

        gr.create_outputs(nval, 2)
        gr.initialise()
        param = 2.5
        gr.set_trueparams([param])

        gr.run(inputs)

        expected = inputs.copy()
        expected[:, 0] = param * expected[:, 0]

        ck1 = np.allclose(gr.outputs, expected)
        ck2 = np.allclose(gr.states[0], np.sum(inputs[:, 0]))

        self.assertTrue(ck1)
        self.assertTrue(ck2)


    def test_calibrate(self):
        gr = Dummy()

        nval = 10
        inputs = np.random.uniform(0, 1, size=(nval, 2))

        param = 2.5
        obs = param * inputs[:, 0]

        gr.create_outputs(nval, 2)
        gr.initialise()

        idx = obs == obs
        gr.calibrate(inputs, obs, idx)

        ck = np.allclose(gr.trueparams[0], param)
        self.assertTrue(ck)


if __name__ == "__main__":
    unittest.main()
