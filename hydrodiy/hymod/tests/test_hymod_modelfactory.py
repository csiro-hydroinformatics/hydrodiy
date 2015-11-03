import os
import re
import unittest

import time

import requests
import tarfile
import numpy as np
import pandas as pd

from hymod import modelfactory

from hymod.models.gr4j import GR4J
from hymod.models.gr2m import GR2M
from hymod.models.turcmezentsev import TurcMezentsev
from hymod.models.lagroute import LagRoute


class ModelFactoryTestCases(unittest.TestCase):


    def setUp(self):
        print('\t=> ModelFactoryTestCase')


    def test_gr2m(self):
        gr = modelfactory.get('gr2m')
        ck = isinstance(gr, GR2M)
        self.assertTrue(ck)


    def test_gr4j(self):
        gr = modelfactory.get('gr4j')
        ck = isinstance(gr, GR4J)
        self.assertTrue(ck)


    def test_lagroute(self):
        gr = modelfactory.get('lagroute')
        ck = isinstance(gr, LagRoute)
        self.assertTrue(ck)


    def test_turc(self):
        gr = modelfactory.get('turcmezentsev')
        ck = isinstance(gr, TurcMezentsev)
        self.assertTrue(ck)



if __name__ == "__main__":
    unittest.main()
