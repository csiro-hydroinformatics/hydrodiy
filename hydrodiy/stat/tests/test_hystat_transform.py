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

        trans = transform.Transform('test', 'a', \
                    mins=[0], defaults=[0.5], maxs=[1],
                    constants=['C1', 'C2'])

        self.assertEqual(trans.name, 'test')
        self.assertEqual(trans.pnames, ['a'])

        # test set and get parameters
        value = 0.5
        trans.params = value
        exp = np.array([value], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params, exp))

        # test set and get parameters
        trans.params = -1
        exp = np.array([trans.mins[0]], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params, exp))

        trans.params = 2
        exp = np.array([trans.maxs[0]], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params, exp))

        # test setitem/getitem
        trans['a'] = 0.5
        self.assertTrue(np.allclose(trans.params, [0.5]))
        self.assertTrue(np.allclose(trans['a'], 0.5))
        self.assertTrue(isinstance(trans['a'], float))

        trans['a'] = 2.
        self.assertTrue(np.allclose(trans.params, trans.maxs))

        trans['a'] = -1
        self.assertTrue(np.allclose(trans.params, trans.mins))

        try:
            trans['a'] = [10, 10]
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('The truth value of'))


        try:
            trans['a'] = np.nan
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Cannot set'))

        try:
            trans.params = [np.nan] * trans.nparams
        except ValueError as err:
            pass
        self.assertTrue(str(err).startswith('Cannot process'))

        # Check constants
        self.assertTrue(np.allclose(trans['C1'], 0.))
        trans['C1'] = 10.
        self.assertTrue(np.allclose(trans['C1'], 10.))


        # Test not implemented methods
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
        self.assertTrue(str(err).startswith('Names are not unique'))


    def test_print(self):
        ''' Test print method '''
        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams
            trans.params = np.random.uniform(-5, 5, size=nparams)
            str(trans)


    def test_forward_backward(self):
        ''' Test if transform is stable when applying it forward then backward '''

        for nm in transform.__all__:

            trans = getattr(transform, nm)()
            nparams = trans.nparams

            for sample in range(500):
                # generate x sample
                x = np.random.normal(size=100, loc=5, scale=20)
                # Generate parameters
                params = trans.mins+np.random.uniform(0., 4, size=nparams)
                params[np.isinf(params)] = 0.
                trans.params = params

                # Handle specific cases
                if nm == 'Log':
                    x = np.exp(x)
                elif nm == 'Logit':
                    x = np.random.uniform(0., 1., size=100)
                    trans.reset()
                elif nm == 'LogSinh':
                    trans._params.values += 0.1
                elif nm == 'Softmax':
                    x = np.random.uniform(0, 1, size=(100, 5))
                    x = x/(0.1+np.sum(x, axis=1)[:, None])

                # Check x -> forward(x) -> backward(y) is stable
                y = trans.forward(x)
                xx = trans.backward(y)

                # Check raw and transform/backtransform are equal
                if x.ndim>1:
                    idx = np.all(~np.isnan(xx) & ~np.isnan(x), axis=xx.ndim-1)
                else:
                    idx = ~np.isnan(xx) & ~np.isnan(x)

                ck = np.allclose(x[idx], xx[idx])
                if not ck:
                    print('\n\n!!! Transform {0} failing the forward/backward test'.format(trans.name))

                self.assertTrue(ck)


    def test_jacobian(self):
        ''' Test of transformation Jacobian is equal to its
        first order numerical approx '''

        delta = 1e-5
        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams

            for sample in range(100):
                x = np.random.normal(size=100, loc=5, scale=20)
                trans.params = trans.mins+np.random.uniform(1e-3, 2, size=nparams)

                if nm in ['Log', 'BoxCox']:
                    x = np.clip(x, 1e-1, np.inf)

                elif nm == 'Logit':
                    x = np.random.uniform(1e-2, 1.-1e-2, size=100)
                    trans.reset()

                elif nm == 'Softmax':
                    x = np.random.uniform(0, 1, size=(100, 5))
                    x = x/(0.1+np.sum(x, axis=1)[:, None])
                    trans.reset()

                #elif nm == 'LogSinh':
                #    trans._params += 0.1

                # Compute first order approx of jac
                y = trans.forward(x)

                if nm == 'Softmax':
                    # A bit more work to compute matrix determinant
                    jacn = np.zeros(x.shape[0])
                    for i in range(x.shape[1]):
                        xd = x[i, :][None, :] + np.eye(x.shape[1])*delta
                        yd = trans.forward(xd)
                        M = (yd-y[i, :][None, :])/delta
                        jacn[i] = np.linalg.det(M)
                else:
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
                ck = np.all(crit[idx]<1e-3)
                if not ck:
                    print('Transform {0} failing the numerical Jacobian test'.format(trans.name))

                self.assertTrue(ck)


    def test_transform_plot(self):
        ''' Plot transform '''
        ftest = self.ftest
        x = np.linspace(-3, 10, 200)

        xs = np.linspace(0, 0.99, 200)
        xs = np.column_stack([xs, (1-xs)*xs])

        for nm in transform.__all__:
            trans = getattr(transform, nm)()
            nparams = trans.nparams

            xx = x
            if nm == 'Softmax':
                xx = xs

            plt.close('all')
            fig, ax = plt.subplots()
            for pp in [-20., 0, 20.]:
                trans.reset()
                trans.params = trans.defaults * (1.+pp/100)
                y = trans.forward(xx)

                xp, yp = xx, y
                if nm == 'Softmax':
                    xp, yp = xx[:, 0], y[:, 0]

                ax.plot(xp, yp, label='params = default {0}% (rp={1})'.format(pp,
                                        trans.params))
            ax.legend(loc=4)
            ax.set_title(nm)
            fig.savefig(os.path.join(ftest, 'transform_'+nm+'.png'))


if __name__ == "__main__":
    unittest.main()
