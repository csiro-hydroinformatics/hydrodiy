import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydata import hyfao

class HyFAOTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> HyFaoTestCase (hydata)')
        self.dt = None # TODO

    def test_get_databases(self):
       
        hyf = hyfao.HyFAO()

        db = hyf.get_databases()

        import pdb; pdb.set_trace()

        self.assertTrue(dcodes.shape == (67, 5))

#    def test_get_domain_codes(self):
#       
#        hyf = hyfao.HyFAO()
#
#        dcodes = hyf.get_domain_codes()
#
#        self.assertTrue(dcodes.shape == (67, 5))
#
#
#    def test_get_object_codes(self):
#       
#        hyf = hyfao.HyFAO()
#
#        for o in hyfao.object_columns:
#            ocodes = hyf.get_object_codes(o, 'GE')
#
#
#    def test_search_object_codes(self):
#       
#        hyf = hyfao.HyFAO()
#    
#        domain_code = 'QC'
#
#        items = hyf.search_object_codes('items', 
#                    domain_code, '(R|r)ice')
#
#        self.assertTrue(items.shape == (1, 4))
#
#
#    def test_get_data(self):
#       
#        hyf = hyfao.HyFAO()
#    
#        domain_code = 'QC'
#
#        items = hyf.search_object_codes('items', 
#                    domain_code, '(R|r)ice').squeeze()
#
#        item_codes = [items['item_code']]
#        item_codes = ['800', '221', '711', '515', '526']
#
#
#        elements = hyf.search_object_codes('elements', 
#                        domain_code, '(P|p)roduction').squeeze()
#        
#        element_codes = [elements['element_code']]
#        element_codes = ['2312', '2413', '2510']
#
#        area_codes = ['5000']
#        area_codes = ['1', '4', '5']
#
#        data = hyf.get_data(
#            domain_code = domain_code,
#            item_codes = item_codes,
#            element_codes = element_codes,
#            area_codes = area_codes,
#            years = ['2012', '2011', '2010'])
#
#        import pdb; pdb.set_trace()


if __name__ == "__main__":
    unittest.main()
