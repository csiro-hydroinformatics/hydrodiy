import os
import unittest
import numpy as np
from hywafari import wutils

import pandas as pd

#try:
#    from wafari import view as w
#    has_wafari = True
#except ImportError:
#    has_wafari = False

has_wafari = False

class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        print('\t=> UtilsTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)
        self.FTEST = FTEST

        self.fxv1 = '/home/magpie/Dropbox/data/test/xvalidate_v1.hdf5'
        self.fxv2 = '/data/nas01/jenkins/jobs/wafari_dm_project/wafari/output/batea_gr4j/murrumbidgee/tinderry/out/xvalidate.hdf5'
        self.ffc = '/data/nas01/jenkins/jobs/wafari_dm_project/wafari/output/batea_gr4j/murrumbidgee/tinderry/out/forecast.hdf5'
        #self.fxv2b = '/home/magpie/Dropbox/data/test/xvalidate_v2b.hdf5'

        self.RUNXVTEST1 = os.path.exists(self.fxv1)
        self.RUNXVTEST2 = os.path.exists(self.fxv2)
        self.RUNFCTEST = os.path.exists(self.ffc)
        #self.RUNXVTEST2b = os.path.exists(self.fxv2b)

    def test_readxv_1(self):
        if self.RUNXVTEST1:
            sim = wutils.readsim_xvalidate(self.fxv1, '410734', 
                    'data|outcomes')
            self.assertTrue(sim.shape==(348, 6201))

    def test_readxv_2(self):
        if self.RUNXVTEST2:
            sim = wutils.readsim_xvalidate(self.fxv2, '410734')
            self.assertTrue(sim.shape==(337, 6641))

    def test_readfc(self):
        if self.RUNFCTEST:
            sim = wutils.read_fc(self.ffc, '410734')
            self.assertTrue(sim.shape==(6640,))

    def test_createproj(self):
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

        if has_wafari:
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

    def test_create_obs(self):
        nval = 1000
        dt = pd.date_range('1980-01-01', freq='D', periods=nval)
        obs = pd.Series(np.random.uniform(0., 100., size=nval), index = dt)
        
        h5file = '%s/streamflow.hdf5'%self.FTEST
        os.system('rm -f %s'%h5file)
        id = '88888'
        wutils.create_obs(h5file, id, 'STREAMFLOW', obs)

        id = '99999'
        wutils.create_obs(h5file, id, 'STREAMFLOW', obs)

    def test_read_obs(self):
        nval = 1000
        dt = pd.date_range('1980-01-01', freq='D', periods=nval)
        obs = pd.Series(np.random.uniform(0., 100., size=nval), index = dt)
        
        h5file = '%s/streamflow.hdf5'%self.FTEST
        os.system('rm -f %s'%h5file)
        id = '88888'
        wutils.create_obs(h5file, id, 'STREAMFLOW', obs)
        data = wutils.read_obs(h5file, id, 'STREAMFLOW', 'daily')

        self.assertTrue(np.allclose(obs.values, data['value'].values))

    def test_create_poama_hindcast(self):
        nval = 20
        nens = 10
        nleadtime = 10

        #nval = 372
        #nens = 166
        #nleadtime = 92

        dt = pd.date_range('1980-01-01', freq='MS', periods=nval)
        forc = None
        for i in range(nens):
            if i%10==0: print('.. generating ens %3d/%3d ..'%(i, nens))
            tmp = pd.DataFrame(np.random.uniform(0, 100, size=(len(dt), nleadtime)))
            tmp.columns = ['lead%2.2d'%k for k in range(nleadtime)]
            tmp['forecast_date'] = dt
            tmp['iens'] = i
            if forc is None: forc = tmp
            else:
                forc = forc.append(tmp, ignore_index=True)

        h5file = '%s/poama_hindcast.hdf5'%self.FTEST
        os.system('rm -f %s'%h5file)
        wutils.create_poama_hindcast(h5file, forc, nleadtime)


    #def test_readxv_2b(self):
    #    if self.RUNXVTEST2b:
    #        sim = wutils.read_xvalidate(self.fxv2b, '922101', 
    #                variable='simulated (STREAMFLOW|outcomes)')
    #        import pdb; pdb.set_trace()
    #        self.assertTrue(sim.shape==(253, 6641))

    

if __name__ == "__main__":
    unittest.main()

