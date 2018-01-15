import os
import re
import unittest
import numpy as np
import datetime
import pandas as pd

import warnings
from requests.exceptions import ReadTimeout

from hydrodiy.data import hyclimind as hyc

class HyClimIndTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyClimIndTestCase (hydata)')


    def test_getdata(self):
        ''' Simple run of getdata. Skip if it times out after 20 sec '''
        for nm in hyc.INDEX_NAMES:
            print('\t\t.. downloading '+nm)
            try:
                data, url = hyc.get_data(nm, timeout=20)
            except ReadTimeout:
                warnings.warn('Requests has timed out')


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
