import os
import time
import unittest
import numpy as np
import pandas as pd

from hydrodiy.data import signatures

class SignaturesTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> SignaturesTestCase (hydata)')

    def test_fdcslope(self):
        ''' Test scalar casts '''
        x = np.linspace(0, 1, 101)
        slp = signatures.fdcslope(x, q1=90, q2=100, cst=0.5)
        #self.assertTrue()

    def test_fdcslope_error(self):
        ''' Test scalar casts '''
        x = np.linspace(0, 1, 101)
        try:
            slp = signatures.fdcslope(x, q1=90, q2=80, cst=0.5)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected q1<q2'))
        else:

        import pdb; pdb.set_trace()



if __name__ == "__main__":
    unittest.main()
