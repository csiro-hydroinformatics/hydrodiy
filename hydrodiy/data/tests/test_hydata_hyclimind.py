import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydrodiy.data import hyclimind

class HyClimIndTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyClimIndTestCase (hydata)')
        self.dt = None # TODO

    def test_getdata(self):

        hyc = hyclimind.HyClimInd()
        names = hyc.index_names

        for n in names:
            d = hyc.get_data(n)


if __name__ == "__main__":
    unittest.main()
