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

    #def test_databases(self):
    #   
    #    hyf = hyfao.HyFAO()

    #    db = hyf.databases()

    #    self.assertTrue('FAOSTAT' in list(db['label']))


    #def test_datasets(self):
    #   
    #    hyf = hyfao.HyFAO()

    #    db = hyf.databases()

    #    #for m in db['mnemonic']:
    #    #    
    #    #    print('gettting datasets from db [%s]' % m)
    #    #    
    #    #    ds = hyf.datasets(database=m)

    #    #    if not ds is None:
    #    #        print('  -> has %d datasets' % ds.shape[0])

    #    #    else:
    #    #        print('  -> no datasets')


    #    ds = hyf.datasets(database='faostat')

    #    self.assertTrue('crop-prod' in list(ds['mnemonic']))


    #def test_dataset_info(self):
    #   
    #    hyf = hyfao.HyFAO()

    #    dims, mems, countries = hyf.dataset_info(
    #            database='faostat', 
    #            dataset='crop-prod')


    def test_dataset_info(self):
          
        hyf = hyfao.HyFAO()
    
        countries = hyf.countries()    


if __name__ == "__main__":
    unittest.main()
