import os
import unittest
import numpy as np
from hywafari import wplots

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




if __name__ == "__main__":
    unittest.main()



