import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydrodiy.data import baseflow
from hydrodiy.io import csv

class BaseflowTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> BaseflowTestCase (hydata)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

    def test_eckhardt(self):
        ''' Test Eckhardt baseflow '''

        fd = os.path.join(self.ftest, 'baseflow_RDF_EC.csv')
        data, _  = csv.read_csv(fd)
        flow = data.iloc[:, 2]
        bflow_expected = data.iloc[:, 3]

        bflow = baseflow.eckhardt(flow, \
                        tau=100,\
                        thresh=0.95, \
                        BFI_max = 0.80)

        self.assertTrue(np.allclose(bflow_expected, bflow))


if __name__ == "__main__":
    unittest.main()
