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
            vect = Vector(['a', 'a'])
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Names are not')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'], defaults=1)
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected vector of length')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=1)
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected vector of length')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=2)
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected vector of length')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=[-1, 0.5])
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected maxs within')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'], defaults=[1]*2, mins=[0]*2, maxs=[0.5]*2)
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected defaults within')))
        else:
            raise Exception('Problem with error generation')


    def test_empty_vector(self):
        for vect in [Vector([]), Vector(None)]:
            self.assertEqual(vect.nval, 0)
            try:
                vect.values = 0
            except ValueError as err:
                self.assertTrue(str(err).startswith(('Expected vector of length')))
            else:
                raise Exception('Problem with error generation')

            dct = vect.to_dict()
            self.assertEqual(dct, {'hitbounds':False, 'data':[], 'nval':0})


    def test_tofromdict(self):
        vect = Vector(['a', 'b'], [0.5]*2, [0]*2, [1]*2)
        dct = vect.to_dict()

        vect2 = Vector.from_dict(dct)
        dct2 = vect2.to_dict()
        self.assertEqual(dct, dct2)


    def test_string(self):
        vect = Vector(['a', 'b'])
        vect.values = [1. ,1.]
        print(vect)


    def test_set_get(self):
        vect = Vector(['a', 'b', 'c'], [0.5]*3, [0]*3, [1]*3)
        values = np.linspace(0, 1, 3)
        expected = np.zeros(3)
        for i, nm in enumerate(vect.names):
            vect[nm] = values[i]
            expected[i] = vect[nm]

        self.assertTrue(np.allclose(values, expected))

        vect['a'] = 10
        self.assertTrue(np.allclose(vect['a'], 1.))

        vect['a'] = -10
        self.assertTrue(np.allclose(vect['a'], 0.))

        try:
            vect['a'] = np.nan
        except ValueError as err:
            self.assertTrue(str(err).startswith('Cannot set value to nan'))
        else:
            raise Exception('Problem with error generation')


    def test_set_get_attributes(self):
        vect = Vector(['a', 'b', 'c'], [0.5]*3, [0]*3, [1]*3)

        vect.a = 0.8
        self.assertTrue(np.allclose(vect.values, [0.8, 0.5, 0.5]))
        self.assertTrue(~vect.hitbounds)

        self.assertTrue(vect.hitbounds)
        vect.a = 2.
        self.assertTrue(np.allclose(vect.values, [1., 0.5, 0.5]))
        self.assertTrue(vect.hitbounds)


    def test_hitbounds(self):
        vect = Vector(['a', 'b'], [0.5]*2, [0]*2, [1]*2)
        self.assertTrue(~vect.hitbounds)

        vect.values = [2]*2
        self.assertTrue(vect.hitbounds)


    def test_values(self):
        try:
            vect = Vector(['a', 'b'])
            vect.values = 1
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Expected vector of length')))
        else:
            raise Exception('Problem with error generation')

        try:
            vect = Vector(['a', 'b'])
            vect.values = [1., np.nan]
        except ValueError as err:
            self.assertTrue(str(err).startswith(('Cannot process value')))
        else:
            raise Exception('Problem with error generation')

        vect = Vector(range(4))
        vect.values = np.arange(4).reshape((2, 2))
        self.assertTrue(len(vect.values.shape) == 1)

        try:
            vect.values = ['a']*4
        except ValueError as err:
            self.assertTrue(str(err).startswith(('could not convert')))
        else:
            raise Exception('Problem with error generation')


    def test_clone(self):
        vect = Vector(['a', 'b'], [0.5]*2, [0]*2, [1]*2)
        vect.values = [0.6]*2
        dct = vect.to_dict()

        vect2 = vect.clone()
        dct2 = vect.to_dict()
        self.assertEqual(dct, dct2)

        vect2['a'] = 1.


    def test_reset(self):
        vect = Vector(['a', 'b'], [0.5]*2, [0]*2, [1]*2)

        vect.values = [0.7]*2
        vect.reset()
        self.assertTrue(np.allclose(vect.values, vect.defaults))



if __name__ == '__main__':
    unittest.main()
