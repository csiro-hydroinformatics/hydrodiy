import os, re
import unittest
import numpy as np
import pandas as pd
import zipfile

from hydrodiy.io import zipjson

class ZipjsonTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> ZipjsonTestCase')
        self.source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(self.source_file)
        self.data = {'key1': 1, 'key2': 'this is a string', \
                        'key3': [0.4, 32.2, 12.45]}

    def test_write_error(self):
        ''' Test zipjson writer errors '''
        filename = os.path.join(self.ftest, 'zipjson_test2.zip')
        source_file = 'bidule'
        comment = 'test'
        try:
            zipjson.write_zipjson(self.data, filename, comment, \
                                source_file, indent=4)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Source file'))
        else:
            raise ValueError('Problem with error handling')

        try:
            zipjson.write_zipjson(self.data, filename+'.bb', comment, \
                                self.source_file, indent=4)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected filename'))
        else:
            raise ValueError('Problem with error handling')


    def test_read_error(self):
        ''' Test zipjson reader error '''
        filename = os.path.join(self.ftest, 'zipjson_error.zip')
        try:
            data, meta = zipjson.read_zipjson(filename)
        except ValueError as err:
            self.assertTrue(str(err).startswith('No data'))
        else:
            raise ValueError('Problem with error handling')


    def test_read(self):
        ''' Test zipjson reader '''
        filename = os.path.join(self.ftest, 'zipjson_test.zip')
        data, meta = zipjson.read_zipjson(filename)
        self.assertEqual(data, self.data)


    def test_write(self):
        ''' Test zipjson writer '''
        filename = os.path.join(self.ftest, 'zipjson_test2.zip')
        comment = 'test'
        zipjson.write_zipjson(self.data, filename, comment, \
                                self.source_file, indent=4)
        data, meta = zipjson.read_zipjson(filename)
        self.assertEqual(data, self.data)
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()
