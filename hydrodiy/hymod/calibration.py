
import math
import random
import time

import numpy as np

from scipy.optimize import fmin_powell as fmin

from hymod.model import Vector, Matrix


def sse(obs, sim, errparams):
    err = obs-sim
    return np.sum(err*err)


def ssqe_bias(obs, sim, errparams):
    err = np.sqrt(obs)-np.sqrt(sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))


def sls_llikelihood(obs, sim, errparams):
    err = obs-sim
    sigma = errparams[0]
    nval = len(obs)

    ll = np.sum(err*err) + sigma + nval * math.log(sigma)
    return ll


class Calibration(object):

    def __init__(self, model, \
            ncalparams, \
            errfun=None, \
            minimize=True, \
            optimizer=fmin, \
            initialise_model=True, \
            timeit=False):

        self._model = model
        self._minimize = minimize
        self._timeit = timeit
        self._ieval = 0
        self._iprint = 0
        self._runtime = np.nan
        self._optimizer = optimizer
        self._initialise_model = initialise_model

        self._observations = None
        self._idx_cal = None

        self._calparams = Vector('calparams', ncalparams)
        self._calparams_means = Vector('calparams_means', ncalparams)
        self._calparams_stdevs = Vector('calparams_stdevs', \
                ncalparams*ncalparams)

        self.errfun = sse


    def __str__(self):
        str = 'Calibration instance for model {0}\n'.format(self._model.name)
        str += '  ncalparams : {0}\n'.format(self.calparams_means.nval)
        str += '  ieval      : {0}\n'.format(self._ieval)
        str += '  runtime    : {0}\n'.format(self._runtime)
        str += '  {0}\n'.format(self.calparams_means)
        str += '  {0}\n'.format(self.calparams)
        str += '  {0}\n'.format(self._model.params)

        return str


    @property
    def calparams(self):
        return self._calparams


    @property
    def calparams_means(self):
        return self._calparams_means


    @property
    def calparams_stdevs(self):
        return self._calparams_stdevs


    @property
    def model(self):
        return self._model


    @property
    def observations(self):
        return self._observations


    @property
    def errfun(self):
        return self._errfun

    @errfun.setter
    def errfun(self, value):
        self._errfun = value

        def objfun(calparams):

            if self._timeit:
                t0 = time.time()

            # Set model parameters
            params = self.cal2true(calparams)
            self._model._params.data = params

            # Exit objectif function if parameters hit bounds
            if np.sum(self._model._params.hitbounds) > 1e-10:
                return np.inf

            # Run model with initialisation if needed
            if self._initialise_model:
                self._model.initialise()

            self._model.run()
            self._ieval += 1

            if self._timeit:
                t1 = time.time()
                self._runtime = (t1-t0)*1000

            # Get error model parameters if they exist
            errparams = self.cal2err(calparams)

            # Compute objectif function
            ofun = self._errfun(self._observations.data[self._idx_cal, :], \
                    self._model._outputs.data[self._idx_cal, :], errparams)

            if not self._minimize:
                ofun *= -1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('Fit {0:3d} : {1:3.3e} {2} ~ {3:.3f} ms'.format( \
                        self._ieval, ofun, self._calparams, \
                        self._runtime))

            return ofun

        self._objfun = objfun


    @property
    def idx_cal(self):
        return self._idx_cal

    @idx_cal.setter
    def idx_cal(self, value):

        if value.dtype == np.dtype('bool'):
            _idx_cal = np.where(value)[0]
        else:
            _idx_cal = value

        if self._observations is None:
            raise ValueError('No observations data. Please allocate')

        if np.max(_idx_cal) >= self._observations.nval:
            raise ValueError('Wrong values in idx_cal')

        self._idx_cal = _idx_cal


    def check(self):
        # Check idx_cal is allocated
        if self._idx_cal is None:
            raise ValueError('No idx_cal data. Please allocate')

        # Check observations are allocated
        if self._observations is None:
            raise ValueError('No observations data. Please allocate')

        # Check inputs are allocated
        if self._model.inputs is None:
            raise ValueError(('No inputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs are initialised
        if np.all(np.isnan(self._model._inputs.data)):
            raise ValueError(('All inputs data are NaN for model {0}.' + \
                ' Please initialise').format(self._model.name))

        # Check outputs are allocated
        if self._model.outputs is None:
            raise ValueError(('No outputs data for model {0}.' + \
                ' Please allocate').format(self._model.name))

        # Check inputs and observations have the right dimension
        n1 = self._model._inputs.nval
        n2 = self._observations.nval
        if n1 != n2:
            raise ValueError(('model.inputs.nval({0}) !=' + \
                ' observations.nval({1})').format(n1, n2))

        n1 = self._model._outputs.nvar
        n2 = self._observations.nvar
        if n1 != n2:
            raise ValueError(('model.outputs.nvar({0}) !=' + \
                ' observations.nvar({1})').format(n1, n2))


    def cal2true(self, calparams):
        return calparams


    def cal2err(self, calparams):
        return None

    def setup(self, observations, inputs):

       self._observations = Matrix.fromdata('observations', observations)

       self._model.allocate(self._observations.nval, self._observations.nvar)
       self._model.inputs.data = inputs


    def sample(self, nsamples, seed=333):

        # Save random state
        random_state = random.getstate()

        # Set seed
        np.random.seed(seed)

        ncalparams = self.calparams_means.nval

        # sample parameters
        samples = np.random.multivariate_normal(\
                self.calparams_means.data, \
                self.calparams_stdevs.data.reshape( \
                    ncalparams, ncalparams), \
                nsamples)

        samples = np.atleast_2d(samples)
        if samples.shape[0] == 1:
            samples = samples.T

        # Reset random state
        random.setstate(random_state)

        return samples


    def explore(self, \
            calparams_explore=None, \
            nsamples = None, \
            iprint=0, \
            seed=333):

        self.check()
        self._iprint = iprint

        if nsamples is None:
            ncalparams = self._calparams_means.nval
            nsamples = int(200 * math.sqrt(ncalparams))

        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        if calparams_explore is None:
            calparams_explore = self.sample(nsamples, seed)
        else:
            calparams_explore = np.atleast_2d(calparams_explore)
            if calparams_explore.shape[0] == 1:
                calparams_explore = calparams_explore.T

        # Systematic exploration
        calparams_best = None

        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = self._objfun(calparams)
            ofun_explore[i] = ofun
            self._ieval += 1

            if self._iprint>0:
                if self._ieval % self._iprint == 0:
                    self._calparams.data = calparams
                    print('Exploration {0}/{1} : {2:3.3e} {3} ~ {4:.2f} ms'.format( \
                        self._ieval, nsamples, ofun, self._calparams, \
                        self._runtime))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_best = calparams

        if calparams_best is None:
            raise ValueError('Could not identify a suitable' + \
                '  parameter by exploration')

        self._calparams.data = calparams_best

        return calparams_best, calparams_explore, ofun_explore


    def fit(self, calparams_ini, iprint=0, *args, **kwargs):

        self.check()
        self._iprint = iprint

        if self._iprint>0:
            ofun_ini = self._objfun(calparams_ini)

            self._calparams.data = calparams_ini
            print('\nFit start: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_ini, self._calparams, self._runtime))

        calparams_final = self._optimizer(self._objfun, \
                    calparams_ini, \
                    disp=self._iprint>0, *args, **kwargs)

        ofun_final = self._objfun(calparams_final)
        outputs_final = self.model.outputs.data

        if self._iprint>0:
            self._calparams.data = calparams_final
            print('\nFit final: {0:3.3e} {1} ~ {2:.2f} ms\n'.format( \
                    ofun_final, self._calparams, self._runtime))

        self._calparams.data = calparams_final

        return calparams_final, outputs_final, ofun_final

