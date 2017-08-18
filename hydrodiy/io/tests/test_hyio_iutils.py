import os, re, json

import unittest

from string import ascii_letters as letters

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

        def test_script(fs, stype='python'):
            ''' Run script and check there are no errors in stderr '''

            # Run system command
            pipe = subprocess.Popen(['python', fs], \
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

            # Get outputs
            stdout, stderr = pipe.communicate()

            # detect errors
            hasError = False
            if len(stderr)>0:
                stderr = str(stderr)
                hasError = bool(re.search('Error', stderr))

            if hasError:
                print('STDERR not null in {0}:\n\t{1}'.format(fs, stderr))

            # If no problem, then remove script
            if not hasError:
                os.remove(fs)

            return stderr, hasError


        # Run defaut script file template
        fs = os.path.join(self.ftest, 'script_test1.py')
        iutils.script_template(fs, 'test')
        stderr, hasError = test_script(fs)
        self.assertFalse(hasError)

        # Run plot script file template
        fs = os.path.join(self.ftest, 'script_test2.py')
        iutils.script_template(fs, 'test', stype='plot')
        stderr, hasError = test_script(fs)
        self.assertFalse(hasError)


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


    def test_str2dict_random_order(self):
        # Generate random keys
        nkeys = 10
        lkeys = 20

        l = [letters[k] for k in range(len(letters))]
        n = ['{0}'.format(k) for k in range(10)]
        l = l+n

        d1 = {}
        for i in range(nkeys):
            key = ''.join(np.random.choice(l, lkeys))
            value = ''.join(np.random.choice(l, lkeys))
            d1[key] = value
        st1 = iutils.dict2str(d1)

        # Select random order of keys
        nrepeat = 100
        for i in range(nrepeat):
            d2 = {}
            keys = np.random.choice(list(d1.keys()), len(d1), replace=False)
            for key in keys:
                d2[key] = d1[key]

            # Generate string and compare with original one
            # This test checks that random perturbation of keys
            # does not affect the string
            st2 = iutils.dict2str(d2)
            self.assertTrue(st1==st2)



    def test_get_logger(self):
        flog1 = os.path.abspath(__file__) + '1.log'
        flog2 = os.path.abspath(__file__) + '2.log'

        # Test error on level
        try:
            logger = iutils.get_logger('bidule', level='INF')
        except ValueError as err:
            self.assertTrue(str(err).startswith('INF not a valid level'))
        else:
            raise Exception('Problem with error handling')

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

        # Close log file handler and delete files
        logger1.handlers[1].close()
        os.remove(flog1)

        logger2.handlers[1].close()
        os.remove(flog2)


    def test_get_ibatch(self):
        idx = iutils.get_ibatch(20, 2, 1)
        self.assertTrue(np.allclose(idx, np.arange(10, 20)))

        try:
            idx = iutils.get_ibatch(20, 40, 1)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Number of sites per batch is 0'))
        else:
            raise Exception('Problem with error handling')


    def test_download(self):
        # File download
        url = 'https://www.google.com'
        fn = os.path.join(self.ftest, 'google.html')
        iutils.download(url, fn)
        with open(fn, 'r', encoding='cp437') as fo:
            txt = fo.read()
        self.assertTrue(txt.startswith('<!doctype html>'))

        # Binary file download
        url = 'https://www.google.com.au/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png'
        fn = os.path.join(self.ftest, 'google.png')
        iutils.download(url, fn)

        # StringIO download - stream
        url = 'https://www.google.com'
        stream = iutils.download(url)
        txt = stream.read().decode('cp437')
        self.assertTrue(txt.startswith('<!doctype html>'))

        # Binary file - stream
        url = 'https://www.google.com.au/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png'
        stream = iutils.download(url)
        bins = stream.read()

        fn = os.path.join(self.ftest, 'google.png')
        with open(fn, 'rb') as fo:
            bins2 = fo.read()
        self.assertTrue(bins==bins2)

        # auth
        url = 'https://httpbin.org/basic-auth/user/pwd'
        stream = iutils.download(url, user='user', pwd='pwd')
        txt = stream.read().decode('cp437')
        js = json.loads(txt)
        self.assertTrue(js['authenticated'])



if __name__ == "__main__":
    unittest.main()
