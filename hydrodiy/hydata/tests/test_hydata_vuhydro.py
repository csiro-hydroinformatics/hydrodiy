import os
import unittest
import numpy as np
import datetime
import pandas as pd

from hydata import evaplib, meteolib

class VuhydroTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> VuhydroTestCase (hydata)')
        self.dt = None # TODO

    def test_ept(self):
        
        e = evaplib.Ept(21.65,67.0,101300.,18200000.,600000.)
        self.assertEqual(6.349456116128078, e)

if __name__ == "__main__":
    unittest.main()
