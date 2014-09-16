import os
import unittest
import numpy as np
from hywafari import wutils

import pandas as pd

try:
    from wafari import view as w
    has_wafari = True
except ImportError:
    has_wafari = False

class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> UtilsTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)
        self.FTEST = FTEST

        self.fxv1 = '/home/magpie/Dropbox/data/test/xvalidate_v1.hdf5'
        self.fxv2 = '/home/magpie/Dropbox/data/test/xvalidate_v2.hdf5'
        #self.fxv2b = '/home/magpie/Dropbox/data/test/xvalidate_v2b.hdf5'

        self.RUNTEST1 = os.path.exists(self.fxv1)
        self.RUNTEST2 = os.path.exists(self.fxv2)
        #self.RUNTEST2b = os.path.exists(self.fxv2b)

    def test_readxv_1(self):
        if self.RUNTEST1:
            sim = wutils.readsim_xvalidate(self.fxv1, '410734', 
                    'data|outcomes')
            self.assertTrue(sim.shape==(348, 6201))

    def test_readxv_2(self):
        if self.RUNTEST2:
            sim = wutils.readsim_xvalidate(self.fxv2, '922101')
            self.assertTrue(sim.shape==(253, 6641))

    def test_createproj(self):
        if has_wafari:

            FW = '%s/wafari_project'%self.FTEST
            project = '%s/wafari'%FW
            if os.path.exists: os.system('rm -rf %s'%project)
            if not os.path.exists(FW): os.mkdir(FW)
            model = 'batea_gr4j'
            
            s1 = {'id':410734, 'name':'tinderry', 'catchment':'tinderry',
                    'description':'Queanbeyan River at Tinderry',
                    'area':490, 'basin':'murrumbidgee', 'drainage':'murray_darling'}
            s2 = {'id':218001, 'name':'tuross_vale', 'catchment':'turossvale',
                    'description':'Tuross River at Tuross Vale',
                    'area':91, 'basin':'tuross', 'drainage':'south_east_coast_nsw'}
            sites = pd.DataFrame([s1, s2])

            wutils.create_project(sites, project, model)

            w.sys.project(FW)
            w.sys.model(model)

            id = '218001'
            basin = 'tuross'
            catchment = 'turossvale'
            w.sys.basin(basin)
            w.sys.catchment(catchment)

            w.hm.newfile()
            w.hm.create(ID=id)
            w.hm.ingest(ID=id, frequency='daily')

            w.sf.newfile()
            w.sf.create(ID=id)
            w.sf.ingest(ID=id, frequency='daily')


    #def test_readxv_2b(self):
    #    if self.RUNTEST2b:
    #        sim = wutils.read_xvalidate(self.fxv2b, '922101', 
    #                variable='simulated (STREAMFLOW|outcomes)')
    #        import pdb; pdb.set_trace()
    #        self.assertTrue(sim.shape==(253, 6641))

    

if __name__ == "__main__":
    unittest.main()

