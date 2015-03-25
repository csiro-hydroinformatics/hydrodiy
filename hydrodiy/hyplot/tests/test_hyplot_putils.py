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

    def test_zoom(self):

        plt.close()
        fig, ax = plt.subplots()    

        nval = 1000
        x = np.random.uniform(0, 1, nval)
        y = np.random.uniform(0, 1, nval)
        ax.plot(x, y, 'o')
        
        xlim0 = ax.get_xlim()
        ylim0 = ax.get_ylim()

        frac = 20

        putils.zoom(ax, frac)

        fp = '%s/zoom.png'%self.FOUT
        fig.savefig(fp)

        xlim1 = ax.get_xlim()
        ylim1 = ax.get_ylim()

        a = float(frac)/100 + 1
        
        x0 = float(xlim0[1]+xlim0[0])/2
        y0 = float(ylim0[1]+ylim0[0])/2

        dx = float(xlim0[1]-xlim0[0])/2
        dy = float(ylim0[1]-ylim0[0])/2

        eps = 1e-10
        self.assertTrue( (abs(xlim1[0]-x0+dx*a)<eps) & (abs(xlim1[1]-x0-dx*a)<eps))
        self.assertTrue( (abs(ylim1[0]-y0+dy*a)<eps) & (abs(ylim1[1]-y0-dy*a)<eps))


if __name__ == "__main__":
    unittest.main()
