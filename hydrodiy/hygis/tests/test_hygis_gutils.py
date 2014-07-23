import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from hygis import gutils
from hygis.oz import Oz

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hygis)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
   

if __name__ == "__main__":
    unittest.main()
