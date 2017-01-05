import os, re, json

import unittest

import pandas as pd
import subprocess
import numpy as np

import matplotlib.pyplot as plt

from hydrodiy.io import iutils, csv

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyio)')
        self.source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(self.source_file)

    def test_find_files(self):
        # Recursive
        folder = '%s/../..' % self.ftest
        pattern = '(_[\\d]{2}){3}( \\(|.txt)'
        found = iutils.find_files(folder, pattern)
        fn = [re.sub('_.*', '', os.path.basename(f))
                                for f in found]
        self.assertTrue(fn == ['findthis']*3)

        # Not recursive
        found = iutils.find_files(folder, pattern, recursive=False)
        self.assertTrue(len(found)==0)

        folder = '%s/find' % self.ftest
        found = iutils.find_files(folder, pattern, recursive=False)
        self.assertTrue(len(found)==3)


    def test_script_template(self):
        sites = pd.DataFrame({'siteid':[1, 2, 3, 4], \
                    'id':['a', 'b', 'c', 'd']})
        fs = os.path.join(self.ftest, 'sites.csv')
        csv.write_csv(sites, fs, 'site list', self.source_file)

        # Run defaut script file template
        fs = os.path.join(self.ftest, 'script_test1.pytest')
        iutils.script_template(fs, 'test')
        pipe = subprocess.Popen(['python', fs],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        if stderr != '':
            print(stderr)
        self.assertTrue(stderr == '')
        os.remove(fs)

        # Run plot script file template
        fs = os.path.join(self.ftest, 'script_test2.pytest')
        iutils.script_template(fs, 'test', stype='plot')
        # (cannot use Popen because matplotlib is throwing warnings)
        subprocess.check_call('python '+fs, shell=True)
        os.remove(fs)

        # Run console script file template
        fs = os.path.join(self.ftest, 'script_test3.pytest')
        iutils.script_template(fs, 'test', stype='console')
        pipe = subprocess.Popen(['python', fs],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        if stderr != '':
            print(stderr)
        self.assertTrue(stderr == '')
        os.remove(fs)

        # Run bash script file template
        fs = os.path.join(self.ftest, 'script_test4.sh')
        iutils.script_template(fs, 'test', stype='bash')
        pipe = subprocess.Popen(fs, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        if stderr != '':
            print(stderr)
        self.assertTrue(stderr == '')
        os.remove(fs)


    def test_str2dict(self):
        prefix = 'this_is_a_prefix'

        data = {'name':'bob', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2)
        self.assertTrue(prefix2 == '')
        self.assertTrue(data == data2)
        self.assertTrue(source == source2)

        source = os.path.join(self.ftest, source + '.csv')
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2)
        self.assertTrue(prefix2 == '')
        self.assertTrue(data == data2)
        self.assertTrue(re.sub('\\.csv', '', \
            os.path.basename(source)) == source2)

        data = {'name':'bob', 'phone':'2010'}
        source = iutils.dict2str(data, prefix=prefix)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == prefix)
        self.assertTrue(source == source2)

        data = {'name':'bob_marley', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == '')
        self.assertTrue(source == source2)

        data = {'name':'bob_marley%$^_12234123', 'phone':'2010'}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        #self.assertTrue(prefix2 == '')
        #self.assertTrue(source == source2)

        data = {'name':'bob', 'phone':2010}
        source = iutils.dict2str(data)
        data2, prefix2 = iutils.str2dict(source, False)
        source2 = iutils.dict2str(data2, prefix2)
        self.assertTrue(data == data2)
        self.assertTrue(prefix2 == '')
        self.assertTrue(source == source2)


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


    def test_download(self):

        # File download
        url = 'https://www.google.com'
        fn = os.path.join(self.ftest, 'google.html')
        iutils.download(url, fn)
        with open(fn, 'r') as fo:
            txt = fo.read()
        self.assertTrue(txt.startswith('<!doctype html>'))

        # StringIO download
        url = 'https://www.google.com'
        txt = iutils.download(url)
        self.assertTrue(txt.startswith('<!doctype html>'))

        # auth
        url = 'https://httpbin.org/basic-auth/user/pwd'
        txt = iutils.download(url, user='user', pwd='pwd')
        js = json.loads(txt)
        self.assertTrue(js['authenticated'])



if __name__ == "__main__":
    unittest.main()
