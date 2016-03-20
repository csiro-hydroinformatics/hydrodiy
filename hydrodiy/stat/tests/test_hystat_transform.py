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

            for sample in range(100):

                x = np.random.normal(size=1000, loc=5, scale=10)
                x = np.exp(x)

                trans.params = np.random.uniform(-5, 5, size=trans.nparams)

                y = trans.forward(x)
                j = trans.jac(x)
                xx = trans.inverse(y)

                ckk1 = np.allclose(x, xx)
                ckk2 = np.all(j>0)

                if not ckk1:
                    import pdb; pdb.set_trace()

                ck1.append(ckk1)
                ck2.append(ckk2)

            self.assertTrue(np.prod(ck1) == 1)
            self.assertTrue(np.prod(ck2) == 1)


if __name__ == "__main__":
    unittest.main()
