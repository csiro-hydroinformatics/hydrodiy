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

        trans = transform.Transform('test', 1)

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


    def test_print(self):
        ''' Test printing for all transforms '''

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh', 'Identity']:

            trans = transform.getinstance(nm)
            if trans.nconstants > 0:
                trans.constants = 5.

            ntparams = trans.ntparams
            trans.tparams = np.random.uniform(-5, 5, size=ntparams)
            str(trans)


    def test_set_params(self):
        ''' Test setting parameters for all transforms '''

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh', 'Identity']:

            trans = transform.getinstance(nm)
            if trans.nconstants > 0:
                trans.constants = 5.

            ntparams = trans.ntparams
            if ntparams == 0:
                continue

            for isample in range(100):
                trans.tparams = np.random.uniform(-5, 5, size=ntparams)

                # Check tparams -> rparams -> tparams works
                tp = trans.tparams.copy()
                rp = trans.rparams.copy()
                trans.rparams = rp
                ck = np.allclose(tp, trans.tparams)
                self.assertTrue(ck)

                trans.tparams = tp
                self.assertTrue(np.allclose(rp, trans.rparams))


    def test_forward_backward(self):
        ''' Test all transforms backward/forward '''

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh', 'Identity']:

            trans = transform.getinstance(nm)
            if trans.nconstants > 0:
                trans.constants = 5.

            for sample in range(100):

                x = np.random.normal(size=1000, loc=5, scale=20)

                if nm == 'Log':
                    x = np.clip(x, -5, np.inf)

                ntparams = trans.ntparams
                trans.tparams = np.random.uniform(-5, 5, size=ntparams)

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                xx = trans.backward(y)

                # Check raw and transform/backtransform are equal
                idx = ~np.isnan(xx)
                ck = np.allclose(x[idx], xx[idx])
                #if not ck:
                #    idx_pb = idx & (~np.allclose(x, xx))
                #    import pdb; pdb.set_trace()
                #    print('Transform {0} failing the forward/backward test'.format(trans.name))
                self.assertTrue(ck)


    def test_jacobian(self):
        ''' Test all transforms jacobians '''

        delta = 1e-5

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh', 'Identity']:

            trans = transform.getinstance(nm)
            if trans.nconstants > 0:
                trans.constants = 5.

            for sample in range(100):

                x = np.random.normal(size=1000, loc=5, scale=20)

                if nm == 'Log':
                    x = np.clip(x, -5, np.inf)

                ntparams = trans.ntparams
                trans.tparams = np.random.uniform(-5, 5, size=ntparams)

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                yp = trans.forward(x+delta)
                jacn =  (yp-y)/delta
                jac = trans.jacobian_det(x)

                # Check jacobian are positive
                idx = ~np.isnan(jac)
                ck = np.all(jac[idx]>0.)
                if not ck:
                    print('Transform {0} failing the positive Jacobian test'.format(trans.name))
                self.assertTrue(ck)

                # Check jacobian are equal
                idx = idx & (jac>0.)
                crit = np.abs(jac-jacn)/(jac+jacn)
                ck = np.all(crit[idx]<1e-2)
                if not ck:
                    print('Transform {0} failing the numerical Jacobian test'.format(trans.name))
                self.assertTrue(ck)


    def test_all_transform_plot(self):

        FOUT = self.FOUT

        for nm in ['Log', 'BoxCox', \
                'YeoJohnson', 'LogSinh']:

            trans = transform.getinstance(nm)

            x = np.linspace(-3, 10, 200)

            ntparams = trans.ntparams

            if nm in ['LogSinh']:
                trans.constants = 5

            plt.close('all')
            fig, ax = plt.subplots()
            for pp in [-5, 0, 2]:
                trans.tparams = [pp] * ntparams
                y = trans.forward(x)
                ax.plot(x, y, label='tparams = {0} (rp={1})'.format(pp,
                                        trans.rparams))
            ax.legend(loc=4)
            ax.set_title(nm)
            fig.savefig(os.path.join(FOUT, 'transform_'+nm+'.png'))



if __name__ == "__main__":
    unittest.main()
