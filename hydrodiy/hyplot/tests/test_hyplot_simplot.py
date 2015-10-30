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


    def test_col2cmap(self):

        colors = {1:'#004C99', 0:'#FF9933', 0.3:'#FF99FF'}

        cmap = putils.col2cmap(colors)

        x = np.arange(1, 257).reshape((1,256))
        fig, ax = plt.subplots()

        ax.pcolor(x, cmap=cmap, vmin=1, vmax=256)

        fp = '%s/cmap.png' % self.FOUT
        fig.savefig(fp)


    def test_line(self):

        fig, ax = plt.subplots()

        nval = 100
        x = np.random.normal(size=nval)
        y = np.random.normal(scale=2, size=nval)

        ax.plot(x, y)

        putils.line(ax, 1, 1, '-')
        putils.line(ax, 1, 0, '--')
        putils.line(ax, 1, np.inf, ':')
        putils.line(ax, 2, np.inf, '-.')

        fp = '%s/line.png' % self.FOUT
        fig.savefig(fp)


if __name__ == "__main__":
    unittest.main()
