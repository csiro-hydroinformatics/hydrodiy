import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydrodiy.data import hyclimind as hyc

class HyClimIndTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyClimIndTestCase (hydata)')

    def test_getdata(self):
        for nm in hyc.INDEX_NAMES:
            data, url = hyc.get_data(nm)

    def test_getdata_error(self):
        try:
            data, url = hyc.get_data('XX')
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Index '))


if __name__ == "__main__":
    unittest.main()
