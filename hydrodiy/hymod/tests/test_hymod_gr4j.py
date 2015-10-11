import os
import re
import unittest

from timeit import Timer
import time

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
from hymod.gr4j import GR4J

class GR4JTestCases(unittest.TestCase):

    def setUp(self):
        print('\t=> GR4JTestCase')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST

    def test_get_calparams_sample(self):
        nsamples = 100
        gr = GR4J()
        samples = gr.get_calparams_sample(nsamples)
        self.assertTrue(samples.shape == (nsamples, 4))

    def test_gr4juh(self):

        gr = GR4J()

        for x4 in np.linspace(0.5, 50, 100):
            gr.set_trueparams([400, -1, 50, x4])

            ck = abs(np.sum(gr.uh)-2) < 1e-5
            if not ck:
                import pdb; pdb.set_trace()
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
        gr.set_states()
        gr.run(inputs)

        out = gr.get_outputs()

        cols = ['Q[mm/d]', 'Ech[mm/d]', 
           'E[mm/d]', 'PR[mm/d]', 
           'Qd[mm/d]', 'Qr[mm/d]',
           'PERC[mm/d]', 'S[mm]', 'R[mm]']

        ck = np.all(out.columns.values.astype(str) == np.array(cols))
        self.assertTrue(ck)
 

    def test_gr4j_detailed(self):
        if not has_gr4j_wafari:
            return

        sites, comment = csv.read_csv('%s/data/sites.csv' % self.FOUT)

        nsites = sites.shape[0]

        #nsites = sites.shape[0]
        #idx = np.arange(0, nsites, 10)
        #sites = sites.iloc[idx, :]

        count = 0
        warmup = 365 * 5

        nsamples = 100
        samples = gr4j.get_paramslib(nsamples)

        gr = gr4j.GR4J()

        gr2 = gr4j_wafari.GR4J()


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
            gr.setoutputs(len(inputs), 1)

            ee = 0.
            bb = 0.
            dta = 0.
            dtb = 0.

            for ip in range(nsamples):

                # First run
                params = samples[ip,:]
                
                # Run
                t0 = time.time()

                gr.setparams(params)
                gr.setstates()
                gr.run(inputs)
                qsim = gr.outputs

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
                e = np.abs(qsim2[warmup:] - qsim[warmup:, 0])
                if np.max(e) > ee:
                    ee = np.max(e) 

                b = np.abs(np.mean(qsim2[warmup:]) - np.mean(qsim[warmup:, 0]))
                b /= np.mean(qsim2[warmup:])
                if b > bb:
                    bb = b


            ta = dta/nsamples
            tb = dtb/nsamples
            print('  Time = %0.2fms(C) ~ %0.2fms (F) /simulation year (%0.1f%%)' % (
                ta, tb, (ta-tb)/tb*100))

            ck = (ee < 2e-4) & (bb < 0.1)

            if not ck:
                print('  failing %s - ee = %f / bb = %f' % (id, ee, bb))

            self.assertTrue(ck)

if __name__ == "__main__":
    unittest.main()
