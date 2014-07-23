import os
import re
import unittest
import numpy as np
import matplotlib.pyplot as plt

from hyio import iutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyio)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_find_files(self):
        folder = '/home/magpie/Dropbox/code/pypackage/hydrodiy' 
        pattern = '(_[\\d]{2}){3}( \\(|.txt)'
        found = iutils.find_files(folder, pattern)
        fn = [re.sub('_.*', '', os.path.basename(f)) 
                                for f in found]
        self.assertTrue(fn == ['findthis']*3)

    def test_extracpat(self):
        regexp = re.compile(r'19[0-9]{2}-[0-9]{2}')
        s = 'I was born on the 1978-023'
        h = iutils.extracpat(s, regexp)
        self.assertTrue(h == '1978-02')
         
if __name__ == "__main__":
    unittest.main()
