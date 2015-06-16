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

    def test_list_databases(self):
       
        hyf = hyfao.HyFAO()

        db = hyf.list_databases()

        self.assertTrue('FAOSTAT' in list(db['label']))


    def test_list_datasets(self):
       
        hyf = hyfao.HyFAO()

        ds = hyf.list_datasets(database='faostat')

        self.assertTrue('crop-prod' in list(ds['mnemonic']))


    def test_dataset_info(self):
       
        hyf = hyfao.HyFAO()

        info = hyf.dataset_info(database='faostat', dataset='crop-prod')


if __name__ == "__main__":
    unittest.main()
