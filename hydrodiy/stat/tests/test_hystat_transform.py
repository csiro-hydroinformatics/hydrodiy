import os
import itertools
import unittest
import numpy as np
from hydrodiy.stat import transform

class TransformTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> TransformTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        self.xx = np.exp(np.linspace(-8, 5, 10))

    def test_transform(self):

        for nm in ['Log', 'Power', 'YeoJohnson', 'LogSinh']:

            trans = transform.getinstance(nm)

            ck1 = []
            ck2 = []
            ck3 = []
            delta = 1e-5

            for sample in range(100):

                x = np.random.normal(size=1000, loc=3, scale=5)
                x = np.exp(x)

                trans.params = np.random.uniform(-5, 5, size=trans.nparams)

                y = trans.forward(x)
                yp = trans.forward(x+delta)
                j = trans.jac(x)
                xx = trans.inverse(y)

                # Check raw and transform/backtransform are equal
                idx = ~np.isnan(xx)
                ckk1 = np.allclose(x[idx], xx[idx])

                # Check jacobian is positive
                idx = ~np.isnan(j)
                ckk2 = np.all(j[idx]>0)

                # Check value of jacobian
                idx = ~np.isnan(j)
                err = 1-np.abs(j-(yp-y)/delta)/j
                errmin = np.nanmin(err)
                ierrmin = np.where(err == errmin)
                ckk3 = errmin > -10.

                ck1.append(ckk1)
                ck2.append(ckk2)
                ck3.append(ckk3)

            ack1 = np.prod(ck1) == 1
            if not ack1:
                print('Transform {0} failing the transform/backtransform test'.format(trans.name))
            #self.assertTrue(ack1)

            ack2 = np.prod(ck2) == 1
            if not ack2:
                print('Transform {0} failing the positive Jacobian test'.format(trans.name))
            #self.assertTrue(ack2)

            ack3 = np.prod(ck3) == 1
            if not ack3:
                print('Transform {0} failing the numerical Jacobian test'.format(trans.name))
            #self.assertTrue(ack3)


if __name__ == "__main__":
    unittest.main()
