import os
import re
import unittest

import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hyio import csv

from hymod.model import ModelError
from hymod.models.gr4j import GR4J
from hymod import errfun


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


    def test_calibrate(self):

        gr = GR4J()
        warmup = 365*5
        count = 1

        fd = '%s/rrtest_%2.2d_timeseries.csv' % (FRR, count)
        d, comment = csv.read_csv(fd, index_col=0, \
                parse_dates=True)
        idx = np.where(d['obs']>=0)
        d = d[np.min(idx)-warmup:]


        fp = '%s/rrtest_%2.2d_grparams.csv' % (FRR, count)
        params, comment = csv.read_csv(fp)

        inputs = d.loc[:, ['rainfall', 'APET']].values
        inputs = np.ascontiguousarray(inputs, np.float64)
        nval = inputs.shape[0]
        idx_cal = np.arange(len(inputs))>=warmup
        idx_cal = np.where(idx_cal)[0]

        # Calibrate
        nval = inputs.shape[0]
        gr.create_outputs(nval, 1)
        
        # Run gr first
        gr.set_trueparams(params['parvalue'])
        gr.initialise()
        gr.run(inputs)
        obs = gr.outputs[:,0].copy()

        # Calibrate on this output
        gr.calibrate(inputs, obs, idx_cal, \
                errfun=errfun.ssqe_bias, \
                iprint=0, \
                timeit=False)


if __name__ == "__main__":
    unittest.main()
