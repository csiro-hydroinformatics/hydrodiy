import os
import unittest
import numpy as np
from hywafari import wplots

from hystat import sutils

import matplotlib.pyplot as plt

import pandas as pd

class WplotTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> WplotTestCase (hywafari)')
        FTEST, testfile = os.path.split(__file__)
        self.FTEST = FTEST

    def test_skillscores(self):
        FIMG = self.FTEST

        scores = (np.random.normal(size=(12,3))+2)*10
        scores = pd.DataFrame(scores)
        scores.columns = ['RMSE', 'CRPS', 'RMSEP']

        fig, ax = plt.subplots()
        wplots.skillscores(scores, ax)
        fp = '%s/skillscores1.png' % FIMG
        fig.savefig(fp)

        scores = (np.random.normal(size=(12,8))+2)*10
        scores = pd.DataFrame(scores)
        scores.columns = ['score%s' % cn for cn in scores.columns]

        fig, ax = plt.subplots()
        wplots.skillscores(scores, ax, ylim=(10, 40), seasonal=False, title='Test')
        fp = '%s/skillscores2.png' % FIMG
        fig.savefig(fp)

    def test_summary(self):
        FIMG = self.FTEST

        scores = (np.random.normal(size=(20,12))+2)*10
        scores = pd.DataFrame(scores)

        fig, axs = plt.subplots(ncols=2)

        desc = ['xxx'] * scores.shape[0]
        pc1 = wplots.summary(scores, axs[0], descriptions = desc)

        reds = plt.get_cmap('Reds')
        pc2 = wplots.summary(scores, axs[1], cmap=reds)

        # Add colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(pc1, cax = cbar_ax)

        fp = '%s/summary.png' % FIMG
        fig.set_size_inches((16,6))
        fig.savefig(fp)

    def test_pit(self):
        FIMG = self.FTEST

        nval = 100
        nens = 1000

        ff = sutils.empfreq(nens)
        forc = np.random.uniform(0, 100, (nval, nens))
        obs = np.random.uniform(0, 100, (nval,))

        fig, ax = plt.subplots()

        wplots.pit(obs, forc, ax)

        fp = '%s/pit.png' % FIMG
        fig.savefig(fp)



if __name__ == "__main__":
    unittest.main()



