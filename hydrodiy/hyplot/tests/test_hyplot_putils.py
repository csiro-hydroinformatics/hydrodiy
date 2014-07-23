import os
import unittest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hyplot import putils
from hygis import oz

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> UtilsTestCase (hyplot)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        
    def test_img2html(self):
        lb = ['MDB', 'Gulf', 'North East Coast']
        url1 = ['urn:bom.gov.au:awris:common:codelist:feature:%s'%id \
                for id in ['dartmouth', 'julius', 'lakedalrymple']] 
        url2 = ['urn:bom.gov.au:awris:common:codelist:feature:%s'%id \
                    for id in ['wyangala', '', 'lakemaraboon']] 
        root_http = 'http://water.bom.gov.au/waterstorage/resources/chart'
        df = pd.DataFrame({'rowlabel':lb, 'url1':url1, 'url2':url2})
        fo = '%s/storage1.html'%self.FOUT
        html = putils.img2html('Storage data online', df, root_http, fo)

        df = pd.DataFrame({'url1':url1, 'url2':url2})
        fo = '%s/storage2.html'%self.FOUT
        html = putils.img2html('Storage data online', df, root_http, fo)

    def test_plot_ensembles(self):
        nens = 100
        nval = 500
        ens = np.random.normal(size=(nval, nens))

        plt.close()
        fig, ax = plt.subplots()    
        ax.plot([0, 500], [-2, 2], 'r', lw=3)
        putils.plot_ensembles(ens, ax)
        ax.legend()
        fp = '%s/ensembles.png'%self.FOUT
        fig.savefig(fp)
        #self.assertTrue(np.array_equal(catval, expected))

if __name__ == "__main__":
    unittest.main()
