import os
import itertools
import unittest
import numpy as np
from hydrodiy.data.containers import Vector
from hydrodiy.stat import transform

import matplotlib
matplotlib.use('Agg')
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

    def test_get_transform(self):
        ''' Test get transform function '''

        for nm in transform.__all__:
            # Get transform
            trans = transform.get_transform(nm)

            # Get transform and set parameters
            nparams = trans.params.nval
            names = trans.params.names
            kwargs = {k:1 for k in names}

            if nm == 'Logit':
                kwargs['upper'] = 2.

            trans = transform.get_transform(nm, **kwargs)

            x = np.linspace(0, 1, 100)
            x /= 1.5*np.sum(x)
            y = trans.forward(x)

            self.assertEqual(nm, trans.name)
            if nparams>0:
                self.assertEqual(1, trans.params.values[0])

    def test_get_transform_errors(self):
        ''' Test errors in get transform '''
        try:
            trans = transform.get_transform('bidule')
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected transform name'))
        else:
            raise ValueError('Problem in error handling')

        # This should work
        trans = transform.get_transform('BoxCox2', lam=1)

        try:
            trans = transform.get_transform('BoxCox2', g=1)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected parameter name'))
        else:
            raise ValueError('Problem in error handling')


    def test_transform_class(self):
        ''' Test the class transform '''

        params = Vector(['a'], [0.5], [0.], [1.])
        constants = Vector(['C1', 'C2'])
        trans = transform.Transform('test', params, constants)

        self.assertEqual(trans.name, 'test')
        self.assertEqual(trans.params.names, ['a'])

        # test set and get parameters
        value = 0.5
        trans.params.values = value
        exp = np.array([value], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params.values, exp))

        # test set and get parameters
        trans.params.values = -1
        exp = np.array([trans.params.mins[0]], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params.values, exp))

        trans.params.values = 2
        exp = np.array([trans.params.maxs[0]], dtype=np.float64)
        self.assertTrue(np.allclose(trans.params.values, exp))

        # test setitem/getitem for individual parameters
        trans.a = 0.5
        self.assertTrue(np.allclose(trans.a, 0.5))
        self.assertTrue(np.allclose(trans.params.values, [0.5]))
        self.assertTrue(np.allclose(trans.params['a'], 0.5))

        trans.params['a'] = 0.5
        self.assertTrue(np.allclose(trans.a, 0.5))
        self.assertTrue(np.allclose(trans.params.values, [0.5]))
        self.assertTrue(np.allclose(trans.params['a'], 0.5))

        trans.a = 2.
        trans.b = 2.
        self.assertTrue(np.allclose(trans.params.values, trans.params.maxs))

        trans.a = -1
        trans.b = -1
        self.assertTrue(np.allclose(trans.params.values, trans.params.mins))

        # test setitem/getitem for individual constants
        trans.C1 = 6
        self.assertTrue(np.allclose(trans.C1, 6))
        self.assertTrue(np.allclose(trans.constants.values, [6, 0.]))
        self.assertTrue(np.allclose(trans.constants['C1'], 6))

        trans.constants['C1'] = 7
        self.assertTrue(np.allclose(trans.C1, 7))
        self.assertTrue(np.allclose(trans.constants.values, [7, 0.0]))
        self.assertTrue(np.allclose(trans.constants['C1'], 7))


    def test_transform_class_error(self):
        ''' Test the error from class transform '''

        params = Vector(['a'], [0.5], [0.], [1.])
        trans = transform.Transform('test', params)

        try:
            trans.a = [10, 10]
        except ValueError as err:
            self.assertTrue(str(err).startswith('The truth value of'))
        else:
            raise Exception('Problem with error handling')

        try:
            trans.a = np.nan
        except ValueError as err:
            self.assertTrue(str(err).startswith('Cannot set'))
        else:
            raise Exception('Problem with error handling')

        try:
            trans.params.values = [np.nan] * trans.params.nval
        except ValueError as err:
            self.assertTrue(str(err).startswith('Cannot process'))
        else:
            raise Exception('Problem with error handling')


    def test_transform_class_methods(self):
        ''' Test the methods of the class transform '''

        params = Vector(['a'], [0.5], [0.], [1.])
        trans = transform.Transform('test', params)

        # Test not implemented methods
        x = np.linspace(0, 1, 10)
        try:
            trans.forward(x)
        except NotImplementedError as err:
            self.assertTrue(str(err).startswith('Method forward'))
        else:
            raise Exception('Problem with error handling')

        try:
            trans.backward(x)
        except NotImplementedError as err:
            self.assertTrue(str(err).startswith('Method backward'))
        else:
            raise Exception('Problem with error handling')

        try:
            trans.jacobian_det(x)
        except NotImplementedError as err:
            self.assertTrue(str(err).startswith('Method jacobian_det'))
        else:
            raise Exception('Problem with error handling')

        try:
            params = Vector(['a', 'a'], [0.5]*2, [0.]*2, [1.]*2)
            trans = transform.Transform('test', params)
        except ValueError as err:
            self.assertTrue(str(err).startswith('Names are not unique'))
        else:
            raise Exception('Problem with error handling')


    def test_print(self):
        ''' Test print method '''
        for nm in transform.__all__:
            trans = transform.get_transform(nm)
            nparams = trans.params.nval
            trans.params.values = np.random.uniform(-5, 5, size=nparams)
            str(trans)


    def test_forward_backward(self):
        ''' Test if transform is stable when applying it forward then backward '''

        for nm in transform.__all__:

            trans = transform.get_transform(nm)
            nparams = trans.params.nval

            for sample in range(500):
                # generate x sample
                x = np.random.normal(size=100, loc=5, scale=20)
                # Generate parameters
                trans.params.values = np.random.uniform(1e-3, 2, size=nparams)

                # Handle specific cases
                if nm == 'Log':
                    x = np.exp(x)
                elif nm == 'Logit':
                    x = np.random.uniform(0., 1., size=100)
                    trans.reset()
                elif nm == 'LogSinh':
                    trans.params.values += 0.1
                    trans.constants.values = 5.
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
                    print(('\n\n!!! Transform {0} failing'+\
                        ' the forward/backward test').format(trans.name))

                self.assertTrue(ck)


    def test_jacobian(self):
        ''' Test of transformation Jacobian is equal to its
        first order numerical approx '''

        delta = 1e-5
        for nm in transform.__all__:
            trans = transform.get_transform(nm)
            nparams = trans.params.nval

            for sample in range(100):
                x = np.random.normal(size=100, loc=5, scale=20)
                trans.params.values = np.random.uniform(1e-3, 2, size=nparams)

                if nm in ['Log', 'BoxCox2']:
                    x = np.clip(x, 1e-1, np.inf)

                elif nm == 'Logit':
                    x = np.random.uniform(1e-2, 1.-1e-2, size=100)
                    trans.reset()

                elif nm == 'Softmax':
                    x = np.random.uniform(0, 1, size=(100, 5))
                    x = x/(0.1+np.sum(x, axis=1)[:, None])
                    trans.reset()

                elif nm == 'LogSinh':
                    trans.constants.values = 5.

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
                    print(('Transform {0} not having'+\
                        ' a strictly positive Jacobian').format(trans.name))
                self.assertTrue(ck)

                # Check jacobian is equal to numerical derivation
                idx = idx & (jac>0.) & (jac<5e3)
                crit = np.abs(jac-jacn)/(1+jac+jacn)
                idx = idx & ~np.isnan(crit)
                ck = np.all(crit[idx]<1e-3)
                if not ck:
                    print(('Transform {0} failing the'+\
                        ' numerical Jacobian test').format(trans.name))

                self.assertTrue(ck)


    def test_transform_censored(self):
        ''' Test censored forward transform '''

        for nm in transform.__all__:
            if nm in ['Softmax']:
                continue

            trans = transform.get_transform(nm)

            for censor in [0.1, 0.5]:
                tcensor = trans.forward(censor)
                xt = np.linspace(tcensor-1, tcensor+1, 100)
                x = trans.backward(xt)

                xc = trans.backward_censored(xt, censor)

                idx = x>censor
                ck = np.allclose(xc[(~idx) & ~np.isnan(xc)], censor)
                self.assertTrue(ck)

                ck = np.allclose(xc[idx], x[idx])
                self.assertTrue(ck)


    def test_transform_plot(self):
        ''' Plot transform '''
        ftest = self.ftest
        x = np.linspace(-3, 10, 200)

        xs = np.linspace(0, 0.99, 200)
        xs = np.column_stack([xs, (1-xs)*xs])

        for nm in transform.__all__:
            trans = transform.get_transform(nm)
            nparams = trans.params.nval

            xx = x
            if nm == 'Softmax':
                xx = xs

            plt.close('all')
            fig, ax = plt.subplots()
            for pp in [-20., 0, 20.]:
                trans.reset()
                trans.params.values = trans.params.defaults * (1.+pp/100)
                y = trans.forward(xx)

                xp, yp = xx, y
                if nm == 'Softmax':
                    xp, yp = xx[:, 0], y[:, 0]

                ax.plot(xp, yp, \
                    label='params = default {0}% (rp={1})'.format(pp,
                            trans.params))
            ax.legend(loc=4)
            ax.set_title(nm)
            fig.savefig(os.path.join(ftest, 'transform_'+nm+'.png'))


    def test_params_samples(self):
        ''' Test parameter sampling '''

        nsamples = 1000

        for nm in transform.__all__:
            if nm == 'Identity':
                continue

            trans = transform.get_transform(nm)
            samples = trans.sample_params(nsamples)
            self.assertEqual(samples.shape, (nsamples, trans.params.nval))

            mins = trans.params.mins
            self.assertTrue(np.all(samples>=mins[None, :]-1e-20))

            maxs = trans.params.maxs
            self.assertTrue(np.all(samples<=maxs[None, :]))

            # Check that all parameters are valid
            for smp in samples:
                trans.params.values = smp


if __name__ == "__main__":
    unittest.main()
