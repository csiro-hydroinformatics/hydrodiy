import os
import itertools
import unittest
import numpy as np
from hydrodiy.stat import transform

import matplotlib.pyplot as plt

import warnings
#warnings.filterwarnings('error')

np.random.seed(0)


class TransformTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> TransformTestCase (hystat)')
        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.xx = np.exp(np.linspace(-8, 5, 10))

    def test_transform_class(self):
        ''' Test the class transform '''

        trans = transform.Transform('test', 'a')
        self.assertEqual(trans.name, 'test')
        self.assertEqual(trans.params_names, ['a'])

        # test set and get parameters
        value = 10
        trans.params = value
        exp = np.array([value], dtype=np.float64)
        self.assertTrue(np.allclose(trans._params, exp))
        self.assertTrue(np.allclose(trans.params, exp))

        # test setitem/getitem
        trans['a'] = 20
        self.assertTrue(np.allclose(trans.params, [20]))
        self.assertTrue(np.allclose(trans['a'], 20))
        self.assertTrue(isinstance(trans['a'], float))

        try:
            trans.params = np.nan
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Cannot set'))

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

        try:
            trans = transform.Transform('test', ['a', 'a'])
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Non unique'))


    def test_print(self):
        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams
            trans.params = np.random.uniform(-5, 5, size=nparams)
            str(trans)


    def test_forward_backward(self):
        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams

            for sample in range(100):
                x = np.random.normal(size=100, loc=5, scale=20)
                trans.params = trans._params_mins+np.random.uniform(0., 2, size=nparams)

                if nm == 'Log':
                    x = np.exp(x)
                elif nm == 'Logit':
                    x = np.random.uniform(0., 1., size=100)
                    trans.reset()
                elif nm == 'LogSinh':
                    trans._params += 0.1

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                xx = trans.backward(y)

                # Check raw and transform/backtransform are equal
                idx = ~np.isnan(xx)
                ck = np.allclose(x[idx], xx[idx])
                if not ck:
                    print('Transform {0} failing the forward/backward test'.format(trans.name))

                self.assertTrue(ck)


    def test_jacobian(self):
        delta = 1e-5
        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams

            for sample in range(100):
                x = np.random.normal(size=100, loc=5, scale=20)
                trans.params = trans._params_mins+np.random.uniform(0., 2, size=nparams)

                if nm in ['Log', 'BoxCox']:
                    x = np.clip(x, 1e-1, np.inf)

                elif nm == 'Logit':
                    x = np.random.uniform(1e-2, 1.-1e-2, size=100)
                    trans.reset()

                #elif nm == 'LogSinh':
                #    trans._params += 0.1

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                yp = trans.forward(x+delta)
                jacn = np.abs(yp-y)/delta
                jac = trans.jacobian_det(x)

                # Check jacobian are positive
                idx = ~np.isnan(jac)
                ck = np.all(jac[idx]>0.)
                if not ck:
                    print('Transform {0} not having Jacobian strictly positive'.format(trans.name))
                self.assertTrue(ck)

                # Check jacobian is equal to numerical derivation
                idx = idx & (jac>0.) & (jac<5e3)
                crit = np.abs(jac-jacn)/(1+jac+jacn)
                idx = idx & ~np.isnan(crit)
                ck = np.all(crit[idx]<5e-4)
                if not ck:
                    print('Transform {0} failing the numerical Jacobian test'.format(trans.name))

                self.assertTrue(ck)


    def test_transform_plot(self):
        ftest = self.ftest
        x = np.linspace(-3, 10, 200)

        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams

            plt.close('all')
            fig, ax = plt.subplots()
            for pp in [-20., 0, 20.]:
                trans.reset()
                trans.params = trans.params_default * (1.+pp/100)
                y = trans.forward(x)
                ax.plot(x, y, label='params = default {0}% (rp={1})'.format(pp,
                                        trans.params))
            ax.legend(loc=4)
            ax.set_title(nm)
            fig.savefig(os.path.join(ftest, 'transform_'+nm+'.png'))



if __name__ == "__main__":
    unittest.main()
