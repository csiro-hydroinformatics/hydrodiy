import os
import unittest
import numpy as np
import pandas as pd

from hyio import minilog

import minilog_fun

class MinilogTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> MinilogTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

        logpath = '%s/log.log' % FTEST
        minilog.logpath = logpath

        if os.path.exists(logpath):
            os.remove(logpath)
        

    def test_log(self):
        
        minilog_fun.foo()

        log = minilog.load()

        self.assertTrue(len(log) == 4)
        

if __name__ == "__main__":
    unittest.main()
