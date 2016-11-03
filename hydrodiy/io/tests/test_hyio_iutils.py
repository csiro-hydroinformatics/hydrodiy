import os
import re
import unittest
import pandas as pd
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from hydrodiy.io import iutils, csv

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyio)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

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
        sites = pd.DataFrame({'siteid':[1, 2, 3, 4], \
                    'id':['a', 'b', 'c', 'd']})
        fs = os.path.join(self.FOUT, 'sites.csv')
        csv.write_csv(sites, fs, 'site list', __file__)
        fs = '%s/script_test1.pytest' % self.FOUT
        iutils.script_template(fs, 'test')
        subprocess.check_call('python ' + fs, shell=True)

        fs = '%s/script_test2.pytest' % self.FOUT
        iutils.script_template(fs, 'test', stype='plot')
        subprocess.check_call('python ' + fs, shell=True)

        fs = '%s/script_test3.pytest' % self.FOUT
        iutils.script_template(fs, 'test', stype='console')
        subprocess.check_call('python ' + fs, shell=True)

        fs = '%s/script_test4.sh' % self.FOUT
        os.remove(fs)
        iutils.script_template(fs, 'test', stype='bash')
        subprocess.check_call(fs, shell=True)


    def test_str2dict(self):
        data = {'name':'bob', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2 = iutils.str2dict(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2 = iutils.str2dict(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob_marley%$^_12234123', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2 = iutils.str2dict(source)
        ck = data == data2
        self.assertTrue(ck)

        data = {'name':'bob', 'phone':2010}
        source = iutils.dict2str(data)
        data2 = iutils.str2dict(source, False)
        ck = data == data2
        self.assertTrue(ck)


    def test_get_logger(self):
        flog1 = os.path.abspath(__file__) + '1.log'
        flog2 = os.path.abspath(__file__) + '2.log'

        # Test error on level
        try:
            logger = iutils.get_logger('bidule', level='INF')
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('INF not a valid level'))

        # Test logging
        logger1 = iutils.get_logger('bidule1', flog=flog1)

        mess = ['flog1 A', 'flog1 B']
        logger1.info(mess[0])
        logger1.info(mess[1])

        self.assertTrue(os.path.exists(flog1))

        with open(flog1, 'r') as fl:
            txt = fl.readlines()
        ck = txt[0].strip().endswith('INFO - '+mess[0])
        ck = ck & txt[1].strip().endswith('INFO - '+mess[1])
        self.assertTrue(ck)

        # Test logging with different format
        logger2 = iutils.get_logger('bidule2',
                        fmt='%(message)s',
                        flog=flog2)

        mess = ['flog2 A', 'flog2 B']
        logger2.warn(mess[0])
        logger2.critical(mess[1])

        self.assertTrue(os.path.exists(flog2))

        with open(flog2, 'r') as fl:
            txt = fl.readlines()
        self.assertEqual(mess, [t.strip() for t in txt])


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
