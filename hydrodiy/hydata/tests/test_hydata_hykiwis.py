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

    def test_getsites(self):
       
        hyk = hykiwis.HyKiwis()

        sites = hyk.get_sites()

        #import pdb; pdb.set_trace()
 
    def test_getattrs(self):
       
        hyk = hykiwis.HyKiwis()

        attrs = hyk.get_tsattrs('410001')
        self.assertEqual(attrs['ts_id'], '54940042')
        self.assertEqual(attrs['ts_unitsymbol'], 'Ml/d')
        self.assertEqual(attrs['station_no'], '410001')
        
        attrs = hyk.get_tsattrs('613002')
        self.assertEqual(attrs['ts_id'], '158791042')
        self.assertEqual(attrs['ts_unitsymbol'], 'Ml/d')
        self.assertEqual(attrs['station_no'], '613002')

    def test_getattrs_all(self):
               
        hyk = hykiwis.HyKiwis()

        sites = hyk.get_sites()

        idx = sites['station_no'].apply(lambda x: x.startswith('41073'))
        sites = sites[idx]

        attrs_all = hyk.get_tsattrs_all(sites)

        self.assertEqual(attrs_all.shape, (6, 8))
 
    def test_getdata(self):
       
        hyk = hykiwis.HyKiwis()

        attrs = hyk.get_tsattrs('410001')
        ts_data = hyk.get_data(attrs, '1980-01-01', '1980-01-05')

        self.assertEqual(list(ts_data.astype(int)), 
                [13964, 14018, 15702, 16402, 15410])

if __name__ == "__main__":
    unittest.main()
