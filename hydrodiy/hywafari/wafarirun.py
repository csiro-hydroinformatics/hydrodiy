#!/usr/bin/env python

import sys
import datetime 
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import os
import json
import re

try:
    from wafari import view as w
    from wafari.model import io
    from wafari.model.forecast import tercile
except ImportError:
    print('Can\'t import wafari\n')
    sys.exit()
    

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tables
from tables.exceptions import NoSuchNodeError
from hyio import csv
from hyio import iutils
from hywafari import wutils
from hydata import dutils

class WafariRun:
    
    def __init__(self,  PROJECT='/ehpdata/jlerat/project_jul_dm', 
                    PROJECTREF='/data/nas01/jenkins/jobs/wafari_dm_project',
                    model = 'batea_gr4j', 
                    freq = 'seasonal', 
                    end_xv = date(2008, 12, 1)):
        
        self.PROJECT = PROJECT
        self.PROJECTREF = PROJECTREF
        self.freq = freq
        self.id = None
        self.model = model
        self.end_xv = end_xv

        # wafari branch
        cmd = 'cd /ehpdata/jlerat/wafari; git symbolic-ref HEAD 1> /ehpdata/jlerat/tmp/br'
        os.system(cmd)
        with open('/ehpdata/jlerat/tmp/br', 'r') as fb:
            self.git_branch = re.sub('.*heads/', '', fb.readline())[:-1]


    def __str__(self):
        str = 'Running wafari using project %s at %s time step\n'%(self.PROJECT, self.freq)
        if not self.id is None:
            str += 'Working with site %s (%s, %s)'%(self.id, self.basin, self.catchment)
        return str

    def _testid(self):
        if self.id is None:
            print('Can\'t do that because self.id is None\nYou need to use WafariRun.set_names first\n')
            return 0
        return 1

    def get_sites(self):
        if self.model=='batea_gr4j':
            js = '%s/wafari/report_dm.json'%self.PROJECT
        else:
            js = '%s/wafari/report.json'%self.PROJECT
        sites = wutils.flattens_json(js)
        return sites 

    def set_names(self, id):
        self.id = id
        sites = self.get_sites()
        self.drainage = sites[sites['id']==self.id]['drainage'][0]
        self.basin = sites[sites['id']==self.id]['basin'][0]
        self.catchment = sites[sites['id']==self.id]['catchment'][0]

        PROJECT = self.PROJECT
        basin = self.basin
        catchment = self.catchment
        model = self.model
         
        FDATA = '%s/wafari/data/%s'%(PROJECT, basin)
        if not os.path.exists(FDATA): 
            os.mkdir(FDATA)
            os.mkdir('%s/meta'%FDATA)
            os.mkdir('%s/tseries'%FDATA)
        self.FDATA = FDATA

        FBASIN = '%s/wafari/output/%s/%s'%(PROJECT, model, basin)
        if not os.path.exists(FBASIN): os.mkdir(FBASIN)
        self.FBASIN = FBASIN
        
        FCATCH = '%s/%s'%(FBASIN, catchment)
        if not os.path.exists(FCATCH): os.mkdir(FCATCH)
        if not os.path.exists('%s/out'%FCATCH): os.mkdir('%s/out'%FCATCH)
        if not os.path.exists('%s/etc'%FCATCH): os.mkdir('%s/etc'%FCATCH)
        self.FCATCH = FCATCH
        
        FPOAMA = '/ehpdata/shared/msdm/data/m24/downscaled'
        if not os.path.exists(FPOAMA): os.mkdir(FPOAMA)
        self.FPOAMA = FPOAMA
        
        FPOAMABASIN = '%s/wafari/output/poama_m24/%s'%(PROJECT, basin)
        if not os.path.exists(FPOAMABASIN): os.mkdir(FPOAMABASIN)
        self.FPOAMABASIN = FPOAMABASIN
        
        FPOAMABASINRT = '%s/wafari/poama_m24_rt/%s'%(PROJECT, basin)
        if not os.path.exists(FPOAMABASINRT): os.mkdir(FPOAMABASINRT)
        if not os.path.exists('%s/%s'%(FPOAMABASIN, catchment)): os.mkdir('%s/%s'%(FPOAMABASIN,catchment))
        if not os.path.exists('%s/%s/out'%(FPOAMABASIN, catchment)): os.mkdir('%s/%s/out'%(FPOAMABASIN,catchment))
        self.FPOAMABASINRT = FPOAMABASINRT

        # Wafari setup
        try:
            w.sys.project(self.PROJECT)
            w.sys.frequency(self.freq)
            w.sys.model(self.model)
            w.sys.basin(self.basin)
            w.sys.catchment(self.catchment)

        except OSError:
            self.overwrite_data()
            self.overwrite_outputs()
            w.sys.project(self.PROJECT)
            w.sys.frequency(self.freq)
            w.sys.model(self.model)
            w.sys.basin(self.basin)
            w.sys.catchment(self.catchment)
            
        info = w.sys.catchment_info()
        for i in range(len(info)):
            if info[i]['AWRC']==self.id: self.area = float(info[i]['area'])

    def overwrite_data(self):
        if self._testid()==0: return None

        PROJECTREF = self.PROJECTREF

        lf = iutils.find_files('%s/wafari/data/%s'%(PROJECTREF, self.basin), 
                                '.*(hdf5$|json$|pkl$|Rdata$)')  
        for f in lf:
            print('.. copying %s to %s\n'%(f, self.PROJECT))
            cmd = 'cp %s %s'%(f, re.sub(self.PROJECTREF, self.PROJECT, f))
            os.system(cmd)

    def overwrite_outputs(self):
        if self._testid()==0: return None

        PROJECTREF = self.PROJECTREF

        lf = iutils.find_files('%s/wafari/output/%s/%s'%(PROJECTREF, self.model, self.basin), 
                                '%s.*(hdf5$|json$|pkl$|Rdata$)'%self.catchment)  
        for f in lf:
            print('.. copying %s to %s\n'%(f, self.PROJECT))
            cmd = 'cp %s %s'%(f, re.sub(self.PROJECTREF, self.PROJECT, f))
            os.system(cmd)

    def overwrite_poama_hindcast(self, PROJECTREF=None):
        if self._testid()==0: return None

        if not PROJECTREF is None:
            self.PROJECTREF = PROJECTREF

        fhind = '%s/%s/out/hindcast.hdf5'%(self.FPOAMABASIN, self.catchment)
        cmd = 'cp %s %s'%(re.sub(self.PROJECT, self.PROJECTREF, fhind), fhind)
        os.system(cmd)


    def ingest_poama_forecast(self):
        if self._testid()==0: return None

        FPOAMASRC = None
        for dd in os.listdir(self.FPOAMA):
            dd2 = os.listdir('%s/%s/average'%(self.FPOAMA, dd))
            for dd3 in dd2:
                se = re.search(self.catchment, dd3)
                if se: 
                    FPOAMASRC = '%s/%s/average/%s'%(self.FPOAMA, dd, dd3)
        assert(FPOAMASRC is not None)           
       
        # Create POAMA RT folder in current PROJECT
        for dd in [self.FPOAMABASINRT, '%s/%s'%(self.FPOAMABASINRT, self.catchment)]:
            if not os.path.exists(dd): os.mkdir(dd)
            
        # Copy POAMA data across
        lf = iutils.find_files(FPOAMASRC, 'csv$')  
        lf2 = [os.path.basename(f) for f in iutils.find_files('%s/%s'%(self.FPOAMABASINRT, self.catchment), 'csv$')] 
        for f in lf:
            if os.path.basename(f) in lf2:
                pass
            else:
                print('   copying %s'%f)
                cmd = 'cp %s %s/%s/%s'%(f, self.FPOAMABASINRT, self.catchment, os.path.basename(f))
                os.system(cmd)

    def ingest_obs(self):
        if self._testid()==0: return None

        # Ingest climate
        id = self.id
        hfile = '%s/wafari/data/%s/tseries/hydromet.hdf5'%(self.PROJECT, self.basin)
        try:
            w.hm.ingest(ID=id, frequency='daily')
            w.hm.ingest(ID=id)
        except NoSuchNodeError:
            print('.. creating node %s in file %s\n'%(id, hfile))
            w.hm.create(ID=id)
            w.hm.ingest(ID=id, frequency='daily')
            w.hm.ingest(ID=id)

        # Check streamflow
        sfile = '%s/wafari/data/%s/tseries/streamflow.hdf5'%(self.PROJECT, self.basin)
        try:
            w.sf.ingest(ID=id, frequency='daily')
            w.sf.ingest(ID=id)
        except NoSuchNodeError:
            print('.. creating node %s in file %s\n'%(id, sfile))
            w.sf.create(ID=id)
            w.sf.ingest(ID=id, frequency='daily')
            w.sf.ingest(ID=id)

    def get_obs(self):
        if self._testid()==0: return None

        try: 
            pe = w.hm.grab(ID=self.id, frequency='daily', variable_type='POTENTIAL_EVAPORATION')
            dt = [datetime.combine(x, datetime.min.time()) for x in pe[0]]
            evap = pd.Series(pe[1], index=dt, name='evap[mm/d]')
            rain = w.hm.grab(ID=self.id, frequency='daily', variable_type='PRECIPITATION')
            dt = [datetime.combine(x, datetime.min.time()) for x in rain[0]]
            rainfall = pd.Series(rain[1], index=dt, name='rainfall[mm/d]')
            q = w.sf.grab(ID=self.id, frequency='daily')
            dt = [datetime.combine(x, datetime.min.time()) for x in q[0]]
            flow = pd.Series(q[1], index=dt, name='q[mm/d]')

        except NoSuchNodeError:
            self.ingest_obs()
            pe = w.hm.grab(ID=self.id, frequency='daily', variable_type='POTENTIAL_EVAPORATION')
            dt = [datetime.combine(x, datetime.min.time()) for x in pe[0]]
            evap = pd.Series(pe[1], index=dt, name='evap[mm/d]')
            rain = w.hm.grab(ID=self.id, frequency='daily', variable_type='PRECIPITATION')
            dt = [datetime.combine(x, datetime.min.time()) for x in rain[0]]
            rainfall = pd.Series(rain[1], index=dt, name='rainfall[mm/d]')
            q = w.sf.grab(ID=self.id, frequency='daily')
            dt = [datetime.combine(x, datetime.min.time()) for x in q[0]]
            flow = pd.Series(q[1], index=dt, name='q[mm/d]')


        return pd.DataFrame({'rainfall[mm/d]':rainfall, 'evap[mm/d]':evap, 'runoff[mm/d]':flow})
  
    def gapfill_streamflow(self):
        if self._testid()==0: return None
        id = self.id

        # Get data
        q = self.grabobs()['runoff[mm/d]']
   
        # Fill up climatology matrix
        qq = pd.DataFrame({'q[mm/d]':q, 'year':np.array([d.year for d in q.index])}, index=q.index)
        y1 = q.index[0].year
        y2 = q.index[-1].year
        qqy = np.zeros((366, y2-y1+1))
        cc = 0
        for y in range(y1, y2+1):
            idx = qq['year']==y
            d1 = qq.index[idx][0]
            i1 = (d1-datetime(d1.year, 1,1)).days
            qqy[i1:(i1+np.sum(idx)), cc] = qq['q[mm/d]'][idx]
            cc = cc+1

        # Pad the end of the series with climatology
        print('.. padding end of  daily streamflow ..')
        end = fc_date+relativedelta(days=5)
        end = datetime(end.year, end.month, end.day)
        start = q.index[-1] + relativedelta(days=1)
 
        day = start
        while day<end:
            iday = (day - datetime(day.year, 1, 1)).days
            idx = range(max(0, iday-10), min(qqy.shape[0]-1, iday+10))
            val = np.median(qqy[idx,:])
            q = q.append(pd.Series([val], index=[day]))
            day = day + relativedelta(days=1)

        # Gapfill daily data
        qq = pd.DataFrame({'q[mm/d]':q.interpolate()}, index=q.index)
        qq['month'] = [datetime(x.year, x.month, 1) for x in qq.index]
        qqm = qq.groupby('month').sum()
        now = datetime.now()
        now = datetime(now.year, now.month, 1)
        qqm['q[mm/d]'][qqm.index==now] = np.nan
        
        # Store filled daily
        print('.. gap filling daily streamflow ..')
        sfile = '%s/wafari/data/%s/tseries/streamflow.hdf5'%(self.PROJECT, self.basin)
        h5 = tables.openFile(sfile, mode='a')
        startgapfill = datetime.now()- relativedelta(years=1)
        for row in h5.getNode('/data/daily/STREAMFLOW/%s'%id):
            dt = datetime.strptime(row['isotime'], '%Y-%m-%d')
            if dt>=startgapfill:
                val = qq[qq.index==dt]['q[mm/d]'].values[0]
                row['value'] = val
                row.update()

        try:
            dt = qq.index[qq.index>dt][0]
            table = h5.getNode('/data/daily/STREAMFLOW/%s'%id)
            row = table.row
            while dt<qq.index[-1]:
                row['isotime'] = '%d-%2.2d-%2.2d'%(dt.year, dt.month, dt.day)
                row['time'] = dt.toordinal()
                row['value'] = qq['q[mm/d]'][qq.index==dt].values[0]
                row['flag'] = 1
                row.append()
                dt = dt+relativedelta(days=1)
            table.flush()
        except IndexError:
            pass


        # Store filled monthly
        print('.. storing gap filling monthly streamflow ..')
        for row in h5.getNode('/data/monthly/STREAMFLOW/%s'%id):
            dt = datetime.strptime(row['isotime'], '%Y-%m-%d')
            if dt>=startgapfill:
                val = qqm[qqm.index==dt]['q[mm/d]'].values[0]
                row['value'] = val
                row.update()

        try:
            dt = qqm.index[qqm.index>dt][0]
            table = h5.getNode('/data/monthly/STREAMFLOW/%s'%self.id)
            row = table.row
            while dt<qqm.index[-1]:
                row['isotime'] = '%d-%2.2d-%2.2d'%(dt.year, dt.month, dt.day)
                row['time'] = dt.toordinal()
                row['value'] = qqm['q[mm/d]'][qqm.index==dt].values[0]
                row['flag'] = 1
                row.append()
                dt = dt+relativedelta(months=1)
            table.flush()
        except IndexError:
            pass

        h5.close()

    #def gapfill_rainfall(self):
        ## Backup rainfall data file
        #cmd = 'cp -f %s %s'%(hfile, re.sub('\\.hdf5', '_beforegapfill.hdf5', hfile))
        #os.system(cmd)
 
        ## Get data
        #p = w.sf.grab(ID=id, frequency='daily')
        #dt = [datetime.combine(x, datetime.min.time()) for x in p[0]]
        #rain = pd.Series(p[1], index=dt, name='rain[mm/d]')
        #
        #print('.. gap filling daily rainfall ..')
        ## Fill daily
        #h5 = tables.openFile(hfile, mode='a')
        #for row in h5.getNode('/data/daily/PRECIPITATION/%s'%id):
        #    dt = datetime.strptime(row['isotime'], '%Y-%m-%d')
        #    if (dt>=start) & (pd.isnull(row['value'])):
        #        row['value'] = 0.
        #        row.update()
        #
        ## Fill monthly
        #print('.. gap filling monthly rainfall ..')
        #for row in h5.getNode('/data/monthly/PRECIPITATION/%s'%id):
        #    dt = datetime.strptime(row['isotime'], '%Y-%m-%d')
        #    if (dt>=start) & (pd.isnull(row['value'])):
        #        row['value'] = 0.
        #        row.update()
        #
        #h5.close()

    def run_xv(self):
        if self._testid()==0: return None
        #w.xv.simulate()
        w.sys.controller.xv.delete_simulations('biascorrect')
        w.xv.bccalibrate()
        w.xv.biascorrect()
        w.xv.newfile(True)
        w.xv.monthlyts()
        w.xv.scores()

    def skillscore_plot(self, FOUT, ylim=[0., 70.], webexport=False):
        if self._testid()==0: return None
        id = self.id
        if not os.path.exists(FOUT): os.mkdir(FOUT)

        # Skill score plot
        plt.close('all')
        w.xv.skillscores(id)
        ax = plt.gca()
        ax.set_ylim(ylim)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_XV_1_%s.png'%(FOUT, self.id, self.freq, self.git_branch)
        if webexport : fn = '%s/%s_XV_1.png'%(FOUT, self.id)
        plt.savefig(fn, dpi=80)
        plt.close()

    def skillscore_plot_month(self, FOUT, month, ylim=None, webexport=False):
        if self._testid()==0: return None
        id = self.id
        freq = self.freq
        version = self.git_branch
        filepathm = '%s/%2.2d'%(FOUT, month)
        if not os.path.exists(filepathm): os.mkdir(filepathm)

        # Produce the two main XV products
        w.xv.tseries(id,month,connect_obs=False)
        ax = plt.gca()
        if not ylim is None:
            ax.set_ylim(ylim)
        fn = '%s/%s_%s_XV_4_%2.2d_%s.png'%(filepathm, id, freq, month, version)

        fig = plt.gcf()
        fig.set_size_inches((8,6))
        plt.savefig(fn, dpi=80)
        plt.close()

        w.xv.pits(id, month, ptype=2)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_XV_6_%2.2d_%s.png'%(filepathm, id, freq, month, version)
        if webexport : fn = '%s/%s_XV_6_%2.2d.png'%(filepathm, id, month)
        plt.savefig(fn, dpi=80)
        plt.close()
           
        # Additional products for web export
        if webexport:
            w.xv.plot(id,month,ptype=1)
            fig = plt.gcf()
            fig.set_size_inches((8,6))
            fn = '%s/%s_%s_XV_2_%2.2d_%s.png'%(filepathm, id, freq, month, version)
            if webexport : fn = '%s/%s_XV_2_%2.2d.png'%(filepathm, id, month)
            plt.savefig(fn, dpi=80)
            plt.close()

            w.xv.pits(id, month, ptype=1)
            fig = plt.gcf()
            fig.set_size_inches((8,6))
            fn = '%s/%s_%s_XV_5_%2.2d_%s.png'%(filepathm, id, freq, month, version)
            if webexport : fn = '%s/%s_XV_5_%2.2d.png'%(filepathm, id, month)
            plt.savefig(fn, dpi=80)
            plt.close()

    def run_fc(self, fc_date = date(2014, 6, 1)):
        if self._testid()==0: return None
        self.fc_date = fc_date
        w.fc.forecast(fc_date)

    def get_fc(self, fc_date = date(2014, 6, 1)):
        if self._testid()==0: return None
        ID = '%s.STREAMFLOW'%self.id
        print('\nGet forecast for %s'%fc_date)

        if fc_date>=self.end_xv:
            print('Forecast comes after end of XV. Now running forecast...\n')
            # Run forecast
            self.run_fc(fc_date)

            # Open forecast file
            ffile = '%s/wafari/output/%s/%s/%s/out/forecast.hdf5'%(self.PROJECT,
                                    self.model, self.basin, self.catchment)
            h5 = tables.openFile(ffile, mode='r')
            fc_data = None
            try:
                fc_data = h5.getNode('/data/forecast', ID, "Array")[:]
                h5.close()
            except NoSuchNodeError:
                print('Error: No node /data/forecast in file %s'%xvfile)
                h5.close()

        else:
            print('Extracting data from XV...\n')
            # Open XV file
            xvfile = '%s/wafari/output/%s/%s/%s/out/xvalidate.hdf5'%(self.PROJECT,
                                    self.model, self.basin, self.catchment)
            h5 = tables.openFile(xvfile, mode='r')
            fc_data = None
            try:
                fc_data_all = h5.getNode('/data/forecast/M%2.2dD01/%s'%(fc_date.month, ID), 'simulation', "Array")[:]
                fc_years = h5.getNode('/data/forecast/M%2.2dD01/%s'%(fc_date.month, ID), 'year', "Array")[:]
                if np.sum(fc_years==fc_date.year)==0:
                    print('Error: No year %d in /data/forecast/M%2.2dD01/%s/year in file %s'%(fc_date.year,
                                fc_date.month, ID, xvfile))
                else:
                    fc_data = fc_data_all[fc_years==fc_date.year].flatten()
                h5.close()
            except NoSuchNodeError:
                print('Error: No node /data/forecast/M%2.2dD01/%s in file %s'%(fc_date.month, ID, xvfile))
                h5.close()

        return fc_data

    def get_fc_ref(self, fc_date = date(2014, 6, 1)):
        if self._testid()==0: return None
        ID = '%s.STREAMFLOW'%self.id
        print('\nGet ref forecast for %s'%fc_date)

        # Open XV file
        xvfile = '%s/wafari/output/%s/%s/%s/out/xvalidate.hdf5'%(self.PROJECT,
                                self.model, self.basin, self.catchment)
        h5 = tables.openFile(xvfile, mode='r')
        fc_ref_data = None
        try:
            fc_ref_data = h5.getNode('/data/forecast/M%2.2dD01/%s'%(fc_date.month, ID), 'reference', "Array")[:]
            h5.close()

        except NoSuchNodeError:
            print('Error: No node /data/forecast/M%2.2dD01/%s in file %s'%(fc_date.month, ID, xvfile))
            h5.close()

        return fc_ref_data

    def get_scores(self, fc_date = date(2014, 6, 1)):
        if self._testid()==0: return None
        ID = '%s.STREAMFLOW'%self.id
        print('\nGet skill scores for %s'%fc_date)

        # Open XV file
        xvfile = '%s/wafari/output/%s/%s/%s/out/xvalidate.hdf5'%(self.PROJECT,
                                self.model, self.basin, self.catchment)
        h5 = tables.openFile(xvfile, mode='r')
        fc_ref_data = None
        try:
            crps = h5.getNode('/data/skillScore/M%2.2dD01/%s'%(fc_date.month, ID), 'CRPS', "Array")[:]
            rmse = h5.getNode('/data/skillScore/M%2.2dD01/%s'%(fc_date.month, ID), 'RMSE', "Array")[:]
            rmsep = h5.getNode('/data/skillScore/M%2.2dD01/%s'%(fc_date.month, ID), 'RMSEP', "Array")[:]
            h5.close()

        except NoSuchNodeError:
            print('Error: No node /data/skillScore/M%2.2dD01/%s in file %s'%(fc_date.month, ID, xvfile))
            h5.close()

        return crps[0], rmse[0], rmsep[0]

    def forecast_plot(self, FOUT, fc_date = date(2014, 6, 1)):
        if self._testid()==0: return None
        id = self.id
        freq = self.freq
        version = self.git_branch
        model = self.model

        filepath = FOUT
        #if webexport:
        #    fcpath = '%s/fc'%FOUT
        #    if not os.path.exists(fcpath): os.mkdir(fcpath)
        #    filepathy = '%s/fc/%d'%(FOUT, fc_date.year)
        #    if not os.path.exists(filepathy): os.mkdir(filepathy)
        #    filepath = '%s/fc/%d/%2.2d'%(FOUT, fc_date.year, fc_date.month)

        if not os.path.exists(filepath): os.mkdir(filepath)

        year_month = '%4.4d_%2.2d'%(fc_date.year, fc_date.month)
        
        plt.close('all')

        w.fc.exprob(id)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_1_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_1_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()
        
        w.fc.hist(id, stat='hd')
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_2_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_2_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()
        
        w.fc.obs(id)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_3_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_3_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()
        
        w.fc.obs(id, stat='exp', onlyhr=False, show_interp=True)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_4_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_4_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()

        w.fc.obs(id, stat='density', onlyhr=False, show_interp=True)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_5_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_5_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()

        w.fc.tercile(id)
        fig = plt.gcf()
        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_6_%s_%s_%s.png'%(filepath, id, freq, year_month, model, version)
        #if webexport : fn = '%s/%s_FC_6_%s.png'%(filepath, id, year_month)
        plt.savefig(fn, dpi=80)
        plt.close()

    def tercile(self, fc_date, FOUT, extension='png', xlim=None,
                    title=None, plot=True, onlyhr=False):

        if self._testid()==0: return None
        id = self.id
        basin = self.basin
        catchment = self.catchment
        freq = self.freq
        PROJECT = self.PROJECT
        version = self.git_branch
        model = self.model

        # Read forecast values.
        Qsim = self.get_fc(fc_date)

        # Get historical reference.
        Qref = self.get_fc_ref(fc_date)

        # Get skill score
        crps, rmse, rmsep = self.get_scores(fc_date)
        skill = rmsep
        skill_desc = 'High'
        if skill<0.: skill_desc = 'No skill'
        if skill<10.: skill_desc = 'Low'

        # Read recent 3 months (aggregated) observed data.
        obs = self.get_obs()
        deltam = relativedelta(months=3)
        if self.freq=='monthly': deltam = relativedelta(months=1)
        start = datetime(fc_date.year, fc_date.month, fc_date.day)
        end = start + deltam -relativedelta(days=1)
        recent_seasonal_Qobs = np.sum(obs['runoff[mm/d]'][(obs.index>=start)&(obs.index<=end)])
        recent_seasonal_Qobs *= self.area*1e-3

        filepath = FOUT
        if not os.path.exists(filepath): os.mkdir(filepath)
        year_month = '%4.4d_%2.2d'%(fc_date.year, fc_date.month)
        
        plt.close('all')

        attributes = {'AWRC':self.id, 'area':self.area, 
            'description':'%s - %s'%(self.basin, self.catchment)}

        tercile.forecast_tercile('TEST',
            fc_date, Qsim, attributes, Qref,
            recent_seasonal_Qobs,
            skill, self.id, title, plot, onlyhr, self.freq,
            skill_desc=skill_desc)

        fig = plt.gcf()
        axs = fig.get_axes()
        if not xlim is None:
            axs[0].set_xlim(xlim)
            axs[1].set_xlim(xlim)
        else:
            xlim = axs[0].get_xlim()

        fig.set_size_inches((8,6))
        fn = '%s/%s_%s_FC_6_%s_%s_%s.%s'%(filepath, id, freq, year_month, model, 
                                version, extension)
        plt.savefig(fn, dpi=80)
        plt.close()

        return xlim
