import os
import re
import unittest
import math

import json

from itertools import product

from timeit import Timer
import time

import pandas as pd
import numpy as np
np.seterr(all='print')

from scipy.special import kolmogorov

from hydrodiy.data.containers import Vector

FHERE = os.path.dirname(os.path.abspath(__file__))

class VectorTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> VectorTestCase')

    def test_init(self):
        # Basic instances
        vect = Vector('a')

        vect = Vector(['a', 'b'])

        try:
            vect = Vector(['a', 'b'], defaults=1)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith(('Expected vector of length')))

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=1)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith(('Expected vector of length')))

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=2)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith(('Expected vector of length')))

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=[-1]*2)
        except ValueError as err:
            pass
        import pdb; pdb.set_trace()
        self.assertTrue(str(err).startswith(('Expected maxs within')))

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=[0.5]*2)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith(('Expected defaults within')))


    def test_fromdict(self):
        pass

    def test_string(self):
        pass

    def test_set_get(self):
        pass

    def test_hitbounds(self):
        pass

    def test_values(self):
        pass

    def test_covar(self):
        pass

    def test_randomise(self):
        pass


    def test_clone(self):
        pass

    def test_to_dict(self):
        pass

if __name__ == '__main__':
    unittest.main()
