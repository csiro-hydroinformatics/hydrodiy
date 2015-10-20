import os
import re
import unittest

import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hyio import csv

has_gr4j_wafari = False
try:
    import gr4j as gr4j_wafari
    has_gr4j_wafari = True
except ImportError:
    pass

from hywafari import wdata
from hymod.models.gr4j import GR4J


import c_hymod_models_utils
UHEPS = c_hymod_models_utils.uh_getuheps()

# Get test data
url_testdata = 'https://drive.google.com/file/d/0B9m81HeozSRzcmNkVmdibEpmMTg'
FOUT = os.path.dirname(os.path.abspath(__file__))
ftar = '%s/rrtests.tar.gz' % FOUT
FRR = re.sub('\\.tar\\.gz', '', ftar)

if not os.path.exists(FRR):
    os.mkdir(FRR)
    req = requests.get(url_testdata, params={'alt':'media'})
    tar = tarfile.open(fileobj=req, mode='r:gz')
    tar.extractall()



class GR4JTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR4JTestCase')
        self.FOUT = FOUT

    def test_print(self):
        gr = GR4J()
        str_gr = '%s' % gr


    def test_get_calparams_sample(self):
        nsamples = 100
        gr = GR4J()
        samples = gr.get_calparams_samples(nsamples)
        self.assertTrue(samples.shape == (nsamples, 4))

    def test_gr4juh(self):
        gr = GR4J()

        for x4 in np.linspace(0, 50, 100):
            gr.set_trueparams([400, -1, 50, x4])

            ck = abs(np.sum(gr.uh)-2) < UHEPS * 2
            self.assertTrue(ck)


    def test_gr4j_dumb(self):
        nval = 1000
        p = np.exp(np.random.normal(0, 2, size=nval))
        pe = np.ones(nval) * 5.
        inputs = np.concatenate([p[:,None], pe[:, None]], axis=1)

        params = [400, -1, 50, 5]

        # Run
        gr = GR4J()
        gr.create_outputs(len(inputs), 9)
        gr.set_trueparams(params)
        gr.initialise()
        gr.run(inputs)

        out = gr.get_outputs()

        cols = ['Q[mm/d]', 'Ech[mm/d]',
           'E[mm/d]', 'Pr[mm/d]',
           'Qd[mm/d]', 'Qr[mm/d]',
           'Perc[mm/d]', 'S[mm]', 'R[mm]']

        ck = np.all(out.columns.values.astype(str) == np.array(cols))
        self.assertTrue(ck)


    def test_gr4j_detailed(self):

        if not has_gr4j_wafari:
            return

        sites, comment = csv.read_csv('%s/data/sites.csv' % self.FOUT)
        nsites = sites.shape[0]

        count = 0
        warmup = 365 * 5

        gr = GR4J()

        gr.trueparams_mins = [10, -5, 1, 0.5]
        gr.trueparams_maxs = [1500, 10, 500, 10]

        gr2 = gr4j_wafari.GR4J()

        nsamples = 100
        samples = np.zeros((nsamples, 4))
        for i in range(nsamples):
            for k in range(4):
                u = np.random.uniform(gr.trueparams_mins[k], gr.trueparams_maxs[k])
                samples[i, k] = u

        for idx, row in sites.iterrows():
            print('\n.. dealing with %3d/%3d ..' % (count, nsites))
            count += 1

            id = row['id']
            d, comment = wdata.get_daily(id)

            idx = np.where(pd.notnull(d['PET']))[0]
            d = d.iloc[idx[0]:idx[-1], :]

            inputs = d.loc[:, ['P', 'PET']].values
            inputs = np.ascontiguousarray(inputs, np.float64)
            nval = inputs.shape[0]
            ny = nval/365

            # Set outputs matrix
            gr.create_outputs(len(inputs), 1)

            ee1 = 0.
            ee2 = 0.
            bb = 0.
            dta = 0.
            dtb = 0.

            for ip in range(nsamples):

                # First run
                params = samples[ip,:]

                # Run
                t0 = time.time()
                
                gr.set_trueparams(params)
                gr.initialise()
                gr.run(inputs)
                qsim = gr.outputs[:,0]

                t1 = time.time()
                dta += 1000 * (t1-t0)

                # Second run
                t0 = time.time()
                gr2.X1 = params[0]
                gr2.X2 = params[1]
                gr2.X3 = params[2]
                gr2.X4 = params[3]

                gr2.init()
                gr2.Sp = params[0]/2
                gr2.Sr = params[2]/2

                qsim2 = gr2.run(inputs[:,0], inputs[:,1])

                t1 = time.time()
                dtb += 1000 * (t1-t0)

                # Comparison
                idx = qsim2[warmup:] < 5.
                if np.sum(idx) > 0:
                    e1 = np.abs(qsim2[warmup:][idx] - qsim[warmup:][idx])
                    if np.max(e1) > ee1:
                        ee1 = np.max(e1)

                idx = qsim2[warmup:] > 5.
                if np.sum(idx) > 0:
                    e2 = np.abs(qsim2[warmup:][idx] - qsim[warmup:][idx])
                    if np.max(e2) > ee2:
                        ee2 = np.max(e2)

                b = np.abs(np.mean(qsim2[warmup:]) - np.mean(qsim[warmup:]))
                b /= np.mean(qsim2[warmup:])
                if b > bb:
                    bb = b


            fact = 1./nsamples/inputs.shape[0]*365.25
            ta = dta * fact
            tb = dtb * fact
            print('  Time = %0.2fms/yr(C) ~ %0.2fms/yr (F) (%0.1f%%)' % (
                ta, tb, (ta-tb)/tb*100))

            ck = (ee1 < 1e-3) & (ee2 < 3e-3) & (bb < 1e-5)

            if not ck:
                print('  failing %s - ee1 = %f / ee2 = %f / bb = %f' % (id, 
                        ee1, ee2, bb))

            self.assertTrue(ck)


    def test_gr4j_detailed2(self):

        warmup = 365 * 5
        gr = GR4J()

        for count in range(1, 11):

            fd = '%s/rrtest_%2.2d_timeseries.csv' % (FRR, count)
            d, comment = csv.read_csv(fd)

            fp = '%s/rrtest_%2.2d_grparams.csv' % (FRR, count)
            params, comment = csv.read_csv(fp)

            inputs = d.loc[:, ['rainfall', 'APET']].values
            inputs = np.ascontiguousarray(inputs, np.float64)
            ny = inputs.shape[0]/365

            # Run gr4j
            gr.create_outputs(len(inputs), 1)

            t0 = time.time()

            gr.set_trueparams(params['parvalue'].values)
            gr.initialise()
            gr.run(inputs)
            qsim = gr.get_outputs().squeeze()

            t1 = time.time()
            dta = 1000 * (t1-t0)
            dta /= len(qsim)/365.25

            # Compare
            idx = np.arange(len(inputs)) > warmup
            expected = d['gr4j'].values[idx]
            err = np.abs(qsim.values[idx] - expected)
            err_thresh = 7e-3
            ck = np.max(err) < err_thresh

            if not ck:
                print(('\t\tTEST %2d : max abs err = '
                    '%0.5f < %0.5f ? %s') % (count, \
                    np.max(err), err_thresh, ck))
            else:
                print('\t\tTEST %2d : max abs err = %0.5f ~ %0.2fms/yr' % ( \
                    count, np.max(err), dta))

            self.assertTrue(ck)


    def test_gr4j_calibrate(self):
        return

        gr = GR4J()
        warmup = 365*5

        for count in range(1, 11):
            fd = '%s/rrtest_%2.2d_timeseries.csv' % (FRR, count)
            d, comment = csv.read_csv(fd, index_col=0, \
                    parse_dates=True)
            idx = np.where(d['obs']>=0)
            d = d[np.min(idx)-warmup:]


            fp = '%s/rrtest_%2.2d_grparams.csv' % (FRR, count)
            params, comment = csv.read_csv(fp)

            inputs = d.loc[:, ['rainfall', 'APET']].values
            inputs = np.ascontiguousarray(inputs, np.float64)
            obs = d.loc[:, 'gr4j'].values
            nval = inputs.shape[0]
            idx_cal = obs>=0
            idx_cal = obs>=0 & (np.arange(len(obs))>=warmup)
            idx_cal = np.where(idx_cal)[0]

            # Calibrate
            nval = inputs.shape[0]
            gr.create_outputs(nval, 1)

            def errfun(obs, sim):
                return np.sum((obs-sim)**2)

            gr.calibrate(inputs, obs, idx_cal, \
                    errfun=errfun, \
                    iprint=0, \
                    timeit=True)

            err = np.abs(gr.trueparams - params['parvalue'])
            ck = np.max(err[[0, 2]]) < 1
            ck = ck & (err[1] < 1e-1)
            ck = ck & (err[3] < 1e-4)

            print('\t\tTEST CALIB %2d : max abs err = %0.5f' % ( \
                    count, np.max(err)))

            if not ck:
                import pdb; pdb.set_trace()

            self.assertTrue(ck)



if __name__ == "__main__":
    unittest.main()
