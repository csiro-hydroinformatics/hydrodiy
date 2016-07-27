import os
import re
import unittest
import numpy as np
import matplotlib.pyplot as plt

from hydrodiy.io import iutils

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyio)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_random_password(self):
        length = 20
        pwd = iutils.random_password(length)
        self.assertTrue(len(pwd)==length)

    def test_find_files(self):
        # Recursive
        folder = '%s/../..' % self.FOUT
        pattern = '(_[\\d]{2}){3}( \\(|.txt)'
        found = iutils.find_files(folder, pattern)
        fn = [re.sub('_.*', '', os.path.basename(f))
                                for f in found]
        self.assertTrue(fn == ['findthis']*3)

        # Not recursive
        found = iutils.find_files(folder, pattern, recursive=False)
        self.assertTrue(len(found)==0)

        folder = '%s/find' % self.FOUT
        found = iutils.find_files(folder, pattern, recursive=False)
        self.assertTrue(len(found)==3)


    def test_script_template(self):
        fs = '%s/script_test1.pytest' % self.FOUT
        iutils.script_template(fs)
        execfile(fs)

        fs = '%s/script_test2.pytest' % self.FOUT
        iutils.script_template(fs, type='plot')
        execfile(fs)


    def test_str2vardict(self):
        data = {'name':'bob', 'phone':2010}
        source = iutils.vardict2str(data)
        data2 = iutils.str2vardict(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley', 'phone':2010}
        source = iutils.vardict2str(data)
        data2 = iutils.str2vardict(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley%$^_12234123', 'phone':2010}
        source = iutils.vardict2str(data)
        data2 = iutils.str2vardict(source)
        ck = data == data2
        self.assertTrue(ck)


    def test_get_logger(self):
        flog = os.path.abspath(__file__) + '.log'

        # Test error on level
        try:
            logger = iutils.get_logger('bidule', level='INF',
                        flog=flog, console=True)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('INF not a valid level'))

        # Test logging
        if os.path.exists(flog): os.remove(flog)
        logger = iutils.get_logger('bidule', level='INFO',
                        flog=flog, console=True)

        for i in range(10):
            logger.info('log '+ str(i))

        self.assertTrue(os.path.exists(flog))



    def test_get_ibatch(self):
        idx = iutils.get_ibatch(20, 2, 1)
        self.assertEqual(idx, range(10, 20))

        try:
            idx = iutils.get_ibatch(20, 40, 1)
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Number of sites per batch is 0'))


if __name__ == "__main__":
    unittest.main()
