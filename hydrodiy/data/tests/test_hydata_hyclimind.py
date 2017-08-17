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
        ''' Simple run of getdata '''
        for nm in hyc.INDEX_NAMES:
            data, url = hyc.get_data(nm)

    def test_getdata_error(self):
        ''' Check error generation for getdata '''
        try:
            data, url = hyc.get_data('XX')
        except ValueError as err:
            self.assertTrue(str(err).startswith('Index '))
        else:
            raise Exception('Problem with error generation')


if __name__ == "__main__":
    unittest.main()
