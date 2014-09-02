import os
import unittest
import numpy as np
from hywafari import wutils

class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> UtilsTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)

        self.fxv1 = '/home/magpie/Dropbox/data/test/xvalidate_v1.hdf5'
        self.fxv2 = '/home/magpie/Dropbox/data/test/xvalidate_v2.hdf5'
        #self.fxv2b = '/home/magpie/Dropbox/data/test/xvalidate_v2b.hdf5'

        self.RUNTEST1 = os.path.exists(self.fxv1)
        self.RUNTEST2 = os.path.exists(self.fxv2)
        #self.RUNTEST2b = os.path.exists(self.fxv2b)

    def test_readxv_1(self):
        if self.RUNTEST1:
            sim = wutils.readsim_xvalidate(self.fxv1, '410734', 
                    'data|outcomes')
            self.assertTrue(sim.shape==(348, 6201))

    def test_readxv_2(self):
        if self.RUNTEST2:
            sim = wutils.readsim_xvalidate(self.fxv2, '922101')
            self.assertTrue(sim.shape==(253, 6641))

    #def test_readxv_2b(self):
    #    if self.RUNTEST2b:
    #        sim = wutils.read_xvalidate(self.fxv2b, '922101', 
    #                variable='simulated (STREAMFLOW|outcomes)')
    #        import pdb; pdb.set_trace()
    #        self.assertTrue(sim.shape==(253, 6641))


if __name__ == "__main__":
    unittest.main()

