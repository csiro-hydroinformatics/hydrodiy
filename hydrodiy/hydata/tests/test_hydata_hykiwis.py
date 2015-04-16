import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydata import hykiwis

class HyKiwisTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyKiwisTestCase (hydata)')
        self.dt = None # TODO

    def test_getattrs(self):
       
        hyk = hykiwis.HyKiwis()

        attrs = hyk.getattrs('410001')
        self.assertEqual(attrs['tsid'], '54940042')
        
        attrs = hyk.getattrs('613002')
        self.assertEqual(attrs['tsid'], '158791042')
 
    def test_getdata(self):
       
        hyk = hykiwis.HyKiwis()

        attrs = hyk.getattrs('410001')
        ts_data = hyk.getdata(attrs, '1980-01-01', '1980-01-05')

        self.assertEqual(list(ts_data.astype(int)), 
                [13964, 14018, 15702, 16402, 15410])

if __name__ == "__main__":
    unittest.main()
