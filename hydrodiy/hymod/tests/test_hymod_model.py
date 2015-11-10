import os
import re
import unittest

from timeit import Timer
import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hyio import csv
from hymod.model import Model, Vector


class Dummy(Model):

    def __init__(self):
        Model.__init__(self, 'dummy', 0, 1, \
            1, 1, \
            ['out'], \
            [-1], \
            [3], \
            [1], \
            [[0.5]])

    def run(self, inputs):
        s0 = self.states[0]
        par0 = self.trueparams[0]
        self.states = np.cumsum(inputs)
        self.outputs = s0 + par0 * self.states


    def cal2true(self):
        xt = self.calparams
        self.trueparams = np.array([np.exp(xt[0])])

 

class ModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ModelTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT

    #def test_get_calparams_sample(self):
    #    nsamples = 100
    #    du = Dummy()
    #    samples = du.get_calparams_samples(nsamples)
    #    self.assertTrue(samples.shape == (nsamples, 1))


    #def test_dummy(self):

    #    nval = 1000
    #    p = np.exp(np.random.normal(0, 2, size=nval))
    #    pe = np.ones(nval) * 5.
    #    inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)

    #    params = [0.5]

    #    # Run
    #    dum = Dummy()
    #    dum.create_outputs(len(inputs), 9)
    #    dum.set_trueparams(params)
    #    dum.initialise(states=[10])
    #    dum.run(inputs)

    #    out = dum.get_outputs()
    #    expected = 10 + params[0] * np.cumsum(inputs)

    #    ck = np.allclose(expected, out.values)
    #    self.assertTrue(ck)

    def test_vector(self):

        v = Vector('test', 3)
        
        v.names = ['a', 'b', 'c']
        try:
            v.names = ['a', 'b']
        except ValueError as e:
            pass
        
        v.min = [-1, 10, 2]
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass

        v.max = [10, 100, 20]
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass


        v.data = [-100, 50, 2]
        try:
            v.data = [5, 3]
        except ValueError as e:
            pass

        self.assertTrue(np.allclose(v.data[0], -1))

        v.data = {'b':30}
        self.assertTrue(np.allclose(v.data[1], 30))
        
        str = '{0}'.format(v)

        import pdb; pdb.set_trace()


if __name__ == "__main__":
    unittest.main()
