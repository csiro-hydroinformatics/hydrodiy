import os
import itertools
import unittest
import numpy as np
from hyio import csv
from hystat import transform

class TransformTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> TransformTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        self.xx = np.exp(np.linspace(-8, 5, 10))

    def test_transform_stationary(self):
        
        for nm in ['Log', 'Power']:
            
            trans = transform.getinstance(nm)

            ck = []

            for sample in range(100):
                x = np.random.normal(size=1000, loc=5, scale=10)
                x = np.exp(x)

                a = np.random.uniform(-5, 5)
                trans.params = [a]

                y = trans.forward(x)
                xx = trans.inverse(y)

                ckk = np.allclose(x, xx)

                ck.append(ckk)

            self.assertTrue(np.prod(ck) == 1)
         
    #def test_yeojonhson_singlevalue(self):
    #    tr = transform.Transform('yeojohnson')
    #    ck = []
    #    sh = np.linspace(0, 5, 10)
    #    for pars in itertools.product(self.xx, sh):
    #        tr.loc = pars[0]
    #        tr.shape = pars[1]
    #        x = np.random.uniform(-1000, 1000)
    #        xt = tr.r2t(x)
    #        xx = tr.t2r(xt)
    #        ck.append(np.allclose(xx, x))

    #    self.assertTrue(np.prod(ck)==1)
    #    
    #def test_yeojonhson_stationary(self):
    #    ck = []
    #    tr = transform.Transform('yeojohnson')
    #    sh = np.linspace(0, 5, 10)
    #    for pars in itertools.product(self.xx, sh):
    #        tr.loc = pars[0]
    #        tr.shape = pars[1]
    #        x = np.linspace(-1000, 1000, 10000)
    #        xt = tr.r2t(x)
    #        xx = tr.t2r(xt)
    #        ck.append(np.allclose(xx, x))

    #    self.assertTrue(np.prod(ck)==1)
    #    
    #def test_logsinh_stationary(self):
    #    tr = transform.Transform('logsinh')
    #    ck = []
    #    for pars in itertools.product(self.xx, self.xx):
    #        tr.loc = pars[0]
    #        tr.scale = pars[1]
    #        x = np.linspace(-tr.loc/tr.scale+1e-2, 50, 10000)
    #        xt = tr.r2t(x)
    #        xx = tr.t2r(xt)
    #        ck.append(np.allclose(xx, x))
    #    self.assertTrue(np.prod(ck)==1)

    #def test_yj_jac(self):
    #    ckn = []
    #    ckpos = []
    #    tr = transform.Transform('yeojohnson')
    #    sh = np.linspace(0, 5, 10)
    #    for pars in itertools.product(self.xx, sh):
    #        tr.loc = pars[0]
    #        tr.shape = pars[1]
    #        x = np.linspace(-1000, 1000, 10000)
    #        j = tr.jac(x)
    #        
    #        # Check jacobian is always positive
    #        ckpos.append(np.prod(j>0))

    #        # Compare with numerical jacobian
    #        dx = 1e-6
    #        xt0 = tr.r2t(x)
    #        xt1 = tr.r2t(x+dx)
    #        jn = (xt1-xt0)/dx 
    #        ckn.append(np.allclose(j, jn))
    #    
    #    self.assertTrue(np.prod(ckpos)==1)
    #    self.assertTrue(np.prod(ckn)==1)
 
    #def test_logsinh_jac(self):
    #    tr = transform.Transform('logsinh')
    #    xx = np.exp(np.linspace(-8, 5, 10))
    #    ckpos = []
    #    ckn = []
    #    for pars in itertools.product(xx, xx):
    #        tr.loc = pars[0]
    #        tr.scale = pars[1]
    #        x = np.linspace(-tr.loc/tr.scale+1e-2, 50, 10000)
    #        w = pars[0]+pars[1]*x
    #        j = tr.jac(x)

    #        # Check jacobian is always positiv
    #        ckpos.append(np.prod(j>0))

    #        # Compare with numerical jacobian
    #        dx = 1e-4
    #        xt0 = tr.r2t(x)
    #        xt1 = tr.r2t(x+dx)
    #        jn = (xt1-xt0)/dx 
    #        idx = w<100
    #        ckn.append(np.allclose(j[idx], jn[idx], rtol=1e-02, atol=1e-02))
    #    
    #    self.assertTrue(np.prod(ckpos)==1)
    #    self.assertTrue(np.prod(ckn)==1)

if __name__ == "__main__":
    unittest.main()
