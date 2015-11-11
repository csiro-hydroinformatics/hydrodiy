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
from hymod.model import Model, Vector, Matrix, Calibration


class Dummy(Model):

    def __init__(self):
        Model.__init__(self, 'dummy',
            nconfig=1,\
            ninputs=2, \
            nparams=2, \
            nstates=2, \
            noutputs_max = 9)

        self.config.names = 'Config1'
        self.states.names = ['State1', 'State2']
        self.params.names = ['Param1', 'Param2']
        self.params.units = ['mm', 'mm']

    def run(self):
        par1 = self.params['Param1']
        S = np.repeat(self.states.data.reshape((1, 2)), self.outputs.nval, 0)

        outputs = par1 * np.cumsum(self.inputs.data, 0) + S
        self.outputs.data = np.append(outputs, \
            np.zeros((self.outputs.nval, self.outputs.nvar-2)), 1)

        self.states.data = self.outputs.data[-1, :2]


class VectorTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> VectorTestCase')

    def test_vector1(self):
        v = Vector('test', 3)
        v.names = ['a', 'b', 'c']
        self.assertTrue(list(v.names) == ['a', 'b', 'c'])
        try:
            v.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _names'))


    def test_vector2(self):
        v = Vector('test', 3)
        v.units = ['m2', 'mm/d', 'GL']
        self.assertTrue(list(v.units) == ['m2', 'mm/d', 'GL'])
        try:
            v.units = ['m2', 'mm/d']
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _units'))


    def test_vector3(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        self.assertTrue(list(v.min.astype(int)) == [-1, 10, 2])
        try:
            v.min = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _min'))


    def test_vector4(self):
        v = Vector('test', 3)
        v.max = [10, 100, 20]
        self.assertTrue(list(v.max.astype(int)) == [10, 100, 20])
        try:
            v.max = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _max'))


    def test_vector5(self):
        v = Vector('test', 3)
        v.min = [-1, 10, 2]
        v.data = [-100, 50, 2]
        self.assertTrue(list(v.data.astype(int)) == [-1, 50, 2])
        try:
            v.data = [5, 3]
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _data'))


    def test_vector6(self):
        v = Vector('test', 3)
        v.names = ['a', 'b', 'c']
        v['b'] = 30
        self.assertTrue(np.allclose(v.data[1], 30))


    def test_vector7(self):
        v = Vector('test', 3)
        str = '{0}'.format(v)


    def test_vector8(self):
        v = Vector('test', 1)
        v.data = 10
        self.assertTrue(v.data.shape == (1, ) and v.data[0] == 10.)


class MatrixTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> MatrixTestCase')

    def test_matrix1(self):
        m1 = Matrix.fromdims('test', 100, 5)
        m1.names = ['a', 'b', 'c', 'd', 'e']
        try:
            m1.names = ['a', 'b']
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _names'))


    def test_matrix2(self):
        m1 = Matrix.fromdims('test', 100, 5)
        d0 = np.random.uniform(0, 1, (100, 5))
        m1.data = d0
        self.assertTrue(m1.data.shape == (100, 5))


    def test_matrix3(self):
        m1 = Matrix.fromdims('test', 100, 5)
        try:
            m1.data = np.random.uniform(0, 1, (10, 5))
        except ValueError as e:
            pass
        self.assertTrue(e.message.startswith('Tried setting _data'))


    def test_matrix4(self):
        m1 = Matrix.fromdims('test', 100, 5)
        str = '{0}'.format(m1)


    def test_matrix5(self):
        m1 = Matrix.fromdims('test', 100, 1)
        m1.data = np.random.uniform(0, 1, 100)
        self.assertTrue(m1.data.shape == (100, 1))


    def test_matrix6(self):
        m1 = Matrix.fromdims('test', 100, 6)
        m2 = Matrix.fromdata('test', m1.data)
        self.assertTrue(m2.data.shape == (100, 6))


    def test_matrix7(self):
        m1 = Matrix.fromdims('test', 100, 5)
        m1.names = ['a', 'b', 'c', 'd', 'e']
        m1['b'] = -2 * np.ones(100)
        m1['e'] = -3 * np.ones(100)
        ms = np.sum(m1.data, 0)
        self.assertTrue(ms[1] == -200 and ms[4] == -300)


class ModelTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> ModelTestCase')
        FOUT = os.path.dirname(os.path.abspath(__file__))
        self.FOUT = FOUT


    def test_model1(self):
        dum = Dummy()
        str = '{0}'.format(dum)


    def test_model2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        dum = Dummy()
        dum.allocate(len(inputs), 9)


    def test_model3(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 9)
        dum.params.data = params


    def test_model4(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 9)
        dum.params.data = params
        dum.initialise(states=[10, 0])
        dum.inputs.data = inputs


    def test_model5(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 9)
        dum.params.data = params
        dum.initialise(states=[10, 0])
        dum.inputs.data = inputs
        dum.run()

        expected1 = 10 + params[0] * np.cumsum(inputs[:, 0])
        ck1 = np.allclose(expected1, dum.outputs.data[:, 0])
        self.assertTrue(ck1)

        expected2 = params[0] * np.cumsum(inputs[:, 1])
        ck2 = np.allclose(expected2, dum.outputs.data[:, 1])
        self.assertTrue(ck2)


class CalibrationTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> CalibrationTestCase')

    def test_calibration1(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 9)
        dum.inputs.data = inputs

        dum.params.data = params
        dum.run()
        observations = dum.outputs.data[:, 0].copy()

        calib = Calibration(dum, \
            2, \
            observations, \
            timeit = True)

        calib.calparams_means.data =  [0, 0]
        calib.calparams_stdevs.data = [1, 0, 0, 1]

        str = '{0}'.format(calib)


    def test_calibration2(self):
        inputs = np.random.uniform(0, 1, (1000, 2))
        params = [0.5, 10.]
        dum = Dummy()
        dum.allocate(len(inputs), 9)
        dum.inputs.data = inputs

        dum.params.data = params
        dum.run()
        observations = dum.outputs.data[:, 0].copy()

        calib = Calibration(dum, \
            2, \
            observations, \
            timeit = True)

        calib.calparams_means.data =  [0, 0]
        calib.calparams_stdevs.data = [1, 0, 0, 1]

        start, explo, explo_ofun = calib.explore(iprint=0)
        final, out, _ = calib.fit(start, iprint=100, ftol=1e-8)

        import pdb; pdb.set_trace()



        self.assertTrue(np.allclose(calib.model.params.data[0], params[0]))


if __name__ == "__main__":
    unittest.main()
