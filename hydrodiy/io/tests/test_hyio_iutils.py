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

    def test_password(self):
        length = 20
        pwd = iutils.password(length)
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


    def test_extracpat(self):
        regexp = re.compile(r'19[0-9]{2}-[0-9]{2}')
        s = 'I was born on the 1978-023'
        h = iutils.extracpat(s, regexp)
        self.assertTrue(h == '1978-02')


    def test_script_template(self):
        fs = '%s/script_test1.pytest' % self.FOUT
        iutils.script_template(fs)
        execfile(fs)

        fs = '%s/script_test2.pytest' % self.FOUT
        iutils.script_template(fs, type='plot')
        execfile(fs)


    def test_find_var(self):
        data = {'name':'bob', 'phone':2010}
        source = iutils.write_var(data)
        data2 = iutils.find_var(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley', 'phone':2010}
        source = iutils.write_var(data)
        data2 = iutils.find_var(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley%$^_12234123', 'phone':2010}
        source = iutils.write_var(data)
        data2 = iutils.find_var(source)
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


if __name__ == "__main__":
    unittest.main()
