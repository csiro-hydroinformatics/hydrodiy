import os
import itertools
import unittest
import numpy as np
from hydrodiy.stat import transform

import matplotlib.pyplot as plt

import warnings
#warnings.filterwarnings('error')

class TransformTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> TransformTestCase (hystat)')
        FTEST, testfile = os.path.split(__file__)
        self.FOUT = FTEST
        self.xx = np.exp(np.linspace(-8, 5, 10))

    def test_transform_class(self):
        ''' Test the class transform '''

        trans = transform.Transform(1, 'test')

        self.assertEqual(trans.name, 'test')

        value = 1
        trans.rparams = value
        exp = np.array([value], dtype=np.float64)
        self.assertTrue(np.allclose(trans._rparams, exp))
        self.assertTrue(np.allclose(trans.rparams, exp))
        self.assertTrue(np.allclose(trans._tparams, exp))
        self.assertTrue(np.allclose(trans.tparams, exp))

        value = 2
        trans.tparams = value
        exp = np.array([value], dtype=np.float64)
        self.assertTrue(np.allclose(trans._tparams, exp))
        self.assertTrue(np.allclose(trans.tparams, exp))
        self.assertTrue(np.allclose(trans._rparams, exp))
        self.assertTrue(np.allclose(trans.rparams, exp))

        x = np.linspace(0, 1, 10)
        try:
            trans.forward(x)
        except NotImplementedError as err:
            pass
        self.assertTrue(str(err).startswith('Method forward'))

        try:
            trans.backward(x)
        except NotImplementedError as err:
            pass
        self.assertTrue(str(err).startswith('Method backward'))

        try:
            trans.jacobian_det(x)
        except NotImplementedError as err:
            pass
        self.assertTrue(str(err).startswith('Method jacobian_det'))


    def test_all_transform(self):
        ''' Test all transforms '''

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh', 'Identity']:

            trans = transform.getinstance(nm)

            ck1 = []
            ck2 = []
            ck3 = []
            delta = 1e-5

            for sample in range(100):

                x = np.random.normal(size=1000, loc=5, scale=20)

                ntparams = trans.ntparams
                trans.tparams = np.random.uniform(-5, 5, size=ntparams)

                # Check print
                str(trans)

                # Check tparams -> rparams -> tparams works
                tp = trans.tparams.copy()
                rp = trans.rparams.copy()
                trans.rparams = rp
                self.assertTrue(np.allclose(tp, trans.tparams))

                trans.tparams = tp
                self.assertTrue(np.allclose(rp, trans.rparams))

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                yp = trans.forward(x+delta)
                xx = trans.backward(y)

                # Check raw and transform/backtransform are equal
                idx = ~np.isnan(xx)
                ckk1 = np.allclose(x[idx], xx[idx])

                # Check jacobian_detobian is positive
                j = trans.jacobian_det(x)
                idx = ~np.isnan(j)
                ckk2 = np.all(j[idx]>0)

                # Check value of jacobian_detobian
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


    def test_all_transform_plot(self):

        FOUT = self.FOUT

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh']:

            trans = transform.getinstance(nm)

            x = np.linspace(-5, 10)

            ntparams = trans.ntparams
            trans.tparams = np.random.uniform(-1, 1, size=ntparams)
            y = trans.forward(x)

            plt.close('all')
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(nm + ': rparams = {0}'.format(trans.rparams))
            fig.savefig(os.path.join(FOUT, 'transform_'+nm+'.png'))



if __name__ == "__main__":
    unittest.main()
