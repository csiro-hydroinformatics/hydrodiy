import os, re
import unittest
import numpy as np
import pandas as pd
import zipfile

from hydrodiy.io import zipjson

class ZipjsonTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> ZipjsonTestCase')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.data = {'key1': 1, 'key2': 'this is a string', 'key3': [0.4, 32.2, 12.45]}

    def test_read(self):
        ''' Test zipjson reader '''
        filename = os.path.join(self.ftest, 'zipjson_test.zip')
        data = zipjson.read_zipjson(filename)
        self.assertEqual(data, self.data)

    def test_write(self):
        ''' Test zipjson writer '''
        filename = os.path.join(self.ftest, 'zipjson_test2.zip')
        zipjson.write_zipjson(self.data, filename, indent=4)
        data = zipjson.read_zipjson(filename)
        self.assertEqual(data, self.data)
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()
