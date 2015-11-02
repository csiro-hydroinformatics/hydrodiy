import math
import random
import time
import numpy as np
import pandas as pd

from scipy.optimize import fmin_powell as fmin

import c_hymod_models_utils


class ModelError(Exception):

    def __init__(self, model, \
            ierr=None, \
            ierr_id=None, \
            message=''):

        self.model = model
        self.message = message

        # Set error messages
        error = np.zeros(50).astype(np.int32)
        c_hymod_models_utils.geterror(error)
        error = { \
                -1 : 'UNKNOWN', \
                error[0] : 'ESIZE_INPUTS', \
                error[1] : 'ESIZE_OUTPUTS', \
                error[2] : 'ESIZE_PARAMS', \
                error[3] : 'ESIZE_STATES', \
                error[4] : 'ESIZE_STATESUH', \
                error[5] : 'ESIZE_CONFIG', \
                error[10] : 'EMODEL_RUN' \
        }


        # Initialise
        if not ierr is None:
            self.ierr = ierr
            if ierr in error:
                self.ierr_id = error[ierr]
            else:
                self.ierr_id = error[-1]

            return

        if not ierr_id is None:
            self.ierr_id = ierr_id
            for ierr in error:
                if error[ierr] == ierr_id:
                    self.ierr = ierr
                    return

            self.ierr = -1
            return

        raise ValueError('Either one of ierr or ierr_id should be' + \
            ' different from None')



    def __str__(self):
        txt = '{0} model : error {1} - {2} : {3}'.format( \
                self.model,
                self.ierr,
                self.ierr_id,
                self.message)

        return repr(txt)



def checklength(x, nx, model, ierr_id, message):
    if len(x) != nx:
        moderr = ModelError(model.name, 
            ierr_id = ierr_id, 
            message='{0}, len(x)({1}) != {2}'.format(message, len(x), nx))
        raise moderr


def vect2txt(x):
    txt = ''
    x = np.atleast_1d(x)
    for i in range(len(x)):
        txt += ' %0.2f' % x[i]
    return txt


class Model(object):

    def __init__(self, name, \
            nconfig, \
            config_names, \
            config_default, \
            ninputs, \
            nuhmaxlength, \
            nstates, \
            ntrueparams, \
            ncalparams, \
            trueparams_names, \
            outputs_names, \
            calparams_means, \
            calparams_stdevs, \
            trueparams_mins, \
            trueparams_maxs, \
            trueparams_default):

        self.name = name
        self.ninputs = ninputs
        self.outputs_names = outputs_names
        self.runtime = np.nan
        self.hitbounds = False

        # Config data
        self.nconfig = nconfig
        self.config_names = config_names
        checklength(self.config_names, nconfig, self, \
                'ESIZE_CONFIG', 'Problem with config_names')

        self.config_default = config_default
        checklength(self.config_default, nconfig, self, \
                'ESIZE_CONFIG', 'Problem with config_default')
        self.set_config_default()

        # UH data
        self.nuhmaxlength = nuhmaxlength
        self.nuhlength = 0
        self.uh = np.zeros(nuhmaxlength).astype(np.float64)
        self.statesuh = np.zeros(nuhmaxlength).astype(np.float64)

        # States data
        self.nstates = nstates
        self.states = np.zeros(nstates).astype(np.float64)

        # Parameter data
        self.ntrueparams = ntrueparams
        self.trueparams = np.ones(ntrueparams) * np.nan

        self.trueparams_names = trueparams_names
        checklength(self.trueparams_names, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_names')

        self.trueparams_default = np.atleast_1d(trueparams_default).astype(np.float64)
        checklength(self.trueparams_default, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_default')

        self.trueparams_mins = np.atleast_1d(trueparams_mins).astype(np.float64)
        checklength(self.trueparams_default, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_mins')

        self.trueparams_maxs = np.atleast_1d(trueparams_maxs).astype(np.float64)
        checklength(self.trueparams_maxs, ntrueparams, self, \
                'ESIZE_PARAMS', 'Problem with trueparams_maxs')

        self.ncalparams = ncalparams
        self.calparams = np.ones(ncalparams) * np.nan

        self.calparams_means = np.atleast_1d(calparams_means).astype(np.float64)
        checklength(self.calparams_means, ncalparams, self, \
                'ESIZE_PARAMS', 'Problem with calparams_means')

        self.calparams_stdevs = np.atleast_2d(calparams_stdevs)
        checklength(self.calparams_stdevs.flat[:], \
                ncalparams*ncalparams, self, \
                'ESIZE_PARAMS', 'Problem with calparams_stdevs')

        self.set_trueparams_default()


    def __str__(self):
        str = '\n{0} model implementation\n'.format(self.name)
        str += '  nconfig  = {0}\n'.format(self.nconfig)
        str += '  nuhlength  = {0}\n'.format(self.nuhlength)
        str += '  nuhmaxlength  = {0}\n'.format(self.nuhmaxlength)
        str += '  nstates = {0}\n'.format(len(self.states))

        if hasattr(self, 'nout'):
            str += '  nout    = {0}\n'.format(self.nout)

        if hasattr(self, 'config'):
            str += '  config  = ['
            for i in range(self.nconfig):
                str += ' {0}:{1:.3f}'.format(self.config_names[i], \
                    self.config[i])
            str += ']\n'

        str += '  calparams  = ['
        for i in range(self.ncalparams):
            str += ' {0:.3f}'.format(self.calparams[i])
        str += ']\n'

        str += '  trueparams = ['
        for i in range(self.ntrueparams):
            str += ' {0}:{1:.3f}'.format(self.trueparams_names[i], \
                self.trueparams[i])
        str += ']\n'

        str += '  states     = ['
        for i in range(self.nstates):
            str += ' {0:.3f}'.format(self.states[i])
        str += ']\n'

        return str


    def create_outputs(self, nval, nout=1):
        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)


    def cal2true(self, calparams):
        ''' No transform by default '''
        return calparams


    def set_uhparams(self):
        pass


    def set_config(self, config):
        self.config = np.atleast_1d(config[:self.nconfig])


    def set_config_default(self):
        self.set_config(self.config_default)


    def set_trueparams_default(self):
        self.set_trueparams(self.trueparams_default)


    def set_trueparams(self, trueparams):
        trueparams = np.atleast_1d(trueparams).astype(np.float64)
        self.trueparams = np.atleast_1d(trueparams[:self.ntrueparams])

        # Check parameters bounds
        self.trueparams = np.clip(self.trueparams, \
            self.trueparams_mins, \
            self.trueparams_maxs)

        self.set_uhparams()


    def set_calparams(self, calparams):
        calparams = np.atleast_1d(calparams)
        self.calparams = np.atleast_1d(calparams[:self.ncalparams])

        trueparams = self.cal2true(self.calparams)
        self.set_trueparams(trueparams)

        self.set_uhparams()


    def initialise(self, states=None, statesuh=None):
        if states is None:
            states = [0.] * self.nstates

        if statesuh is None:
            statesuh = [0.] * self.nuhmaxlength

        states = np.atleast_1d(states).astype(np.float64)
        states = np.atleast_1d(states[:self.nstates])
        self.states[:self.nstates] = states

        statesuh = np.array(statesuh).astype(np.float64)
        statesuh = np.atleast_1d(statesuh[:self.nuhlength])
        self.statesuh[:self.nuhlength] = statesuh


    def run(self, inputs):
        pass


    def get_outputs(self):
        outputs = pd.DataFrame(self.outputs)
        outputs.columns = self.outputs_names[:self.nout]

        return outputs


    def get_calparams_samples(self, nsamples, seed=333):

        # Save random state
        random_state = random.getstate()

        # Set seed
        np.random.seed(seed)

        # sample parameters
        samples = np.random.multivariate_normal(self.calparams_means, \
                self.calparams_stdevs, \
                nsamples)

        samples = np.atleast_2d(samples)

        if samples.shape[0] == 1:
            samples = samples.T

        # Reset random state
        random.setstate(random_state)

        return samples


    def calibrate(self, inputs, observations,
            idx_cal=None, \
            errfun=None, \
            nsamples=None,\
            calparams_explore = None, \
            iprint=10, \
            minimize=True, \
            timeit=False):
        ''' Perform model calibration

        Parameters
        -----------
        inputs : numpy.ndarray
            Model input over the calibration period
        obs : numpy.ndarray
            Observation to be matched by model output over the calibration period
        idx_cal : numpy.ndarray
            Indexes to be used for computing the objective function.
            idx_cal is useful when performing split sample tests or cross validation.
            Default considers all data points.
        errfun : function
            Objective function to be minimised or maximised during calibration.
            Arguments of errfun should be:
            - obs : observed data
            - sim : model output to match observed data
            Function should return a float.
            Default is the sum of squared errors.
        nsamples : int
            Number of parameter samples used for initial pre-filtering of parameter sets
            Default is 200xsqrt(ncalparams)
        calparams_explore : numpy.ndarray
            Parameter library used for initial pre-filtering
            Default is sampled from self.get_calparams_samples
        iprint : int
            Prints detailed log every [iprint] iterations
        minimize : bool
            Indicate if errfun should be minimised (True) or maximised (False).
        timeit : bool
            Indicate if the log should contain model runtime

        Returns
        -----------
        calparams_final : numpy.ndarray
            Optimal parameter set
        outputs_final : numpy.ndarray
            Model output generated with the optimal parameter set
        ofun_final : float
            Value of the objective function corresponding to the optimal parameter set
        calparams_explore : numpy.ndarray
            Parameter sets used for the pre-filtering
        ofun_explore : numpy.ndarray
            Objective function values corresponding to the pre-filtering parameter sets.

        Example
        -----------
        TODO
        >>> dutils.normaliseid('00a04567B.100')
        'A04567'

        '''

        # Set defaults
        if idx_cal is None:
            idx_cal = observations == observations

        if errfun is None:
            def fun(obs, sim): return np.sum((obs-sim)**2)
            errfun = fun

        if nsamples is None:
            nsamples = int(200*math.sqrt(self.ncalparams))

        if calparams_explore is None:
            calparams_explore = self.get_calparams_samples(nsamples)
        else:
            calparams_explore = np.atleast_2d(calparams_explore)
            if calparams_explore.shape[0] == 1:
                calparams_explore = calparam_explore.T
            nsamples = calparams_explore.shape[0]

        # check inputs
        if idx_cal.dtype == np.dtype('bool'):
            idx_cal = np.where(idx_cal)[0]

        if len(observations.shape) <= 1:
            observations = np.atleast_2d(observations).T

        if len(inputs.shape) <= 1:
            inputs = np.atleast_2d(inputs).T

        nvalobs = observations.shape[0]
        nvalinputs = inputs.shape[0]
        nobs = observations.shape[1]
        noutputs = nobs

        if nvalobs != nvalinputs:
            raise ModelError(self.name,
                ierr_id='ESIZE_INPUTS',
                message = \
                'model.calibrate, nvalobs({0}) != nvalinputs({1})'.format( \
                nvalobs, nvalinputs))

        if np.max(idx_cal) >= nvalobs:
            raise ModelError(self.name,
                ierr_id='ESIZE_INPUTS',
                message = \
                'model.calibrate, np.max(idx_cal)({0}) >= nvalobs({1})'.format( \
                np.max(idx_cal), nvalobs))

        # Allocate memory
        self.create_outputs(nvalobs, noutputs)

        # Define objective function
        def objfun(calparams):

            if timeit:
                t0 = time.time()

            self.set_calparams(calparams)
            self.initialise()
            self.run(inputs)

            if timeit:
                t1 = time.time()
                self.runtime = 1000 * (t1-t0)

            if noutputs > 1:
                ofun = errfun(observations[idx_cal, :], \
                    self.outputs[idx_cal, :])
            else:
                ofun = errfun(observations[idx_cal], \
                    self.outputs[idx_cal])

            if not minimize:
                ofun *= -1

            return ofun


        # Systematic exploration
        ofun_explore = np.zeros(nsamples) * np.nan
        ofun_min = np.inf

        calparams_ini = []
        for i in range(nsamples):
            calparams = calparams_explore[i,:]
            ofun = objfun(calparams)
            ofun_explore[i] = ofun

            if iprint>0:
                if i%iprint == 0:
                    print('Exploration %d/%d : %03.3e [%s] ~ %0.2f ms' % ( \
                        i, nsamples, ofun, vect2txt(calparams), \
                        self.runtime))

            if ofun < ofun_min:
                ofun_min = ofun
                calparams_ini = calparams

        # Minisation
        if iprint>0:
            print('\nOptimization start: %03.3e [%s] ~ %0.2f ms\n' % ( \
                    ofun_min, vect2txt(calparams_ini), self.runtime))

        calparams_final = fmin(objfun, calparams_ini, disp=iprint>0)
        ofun_final = objfun(calparams_final)
        outputs_final = self.outputs

        if iprint>0:
            print('\nOptimization end: %03.3e [%s] ~ %0.2f ms\n' % ( \
                    ofun_final, vect2txt(calparams_final), self.runtime))

        return calparams_final, outputs_final, ofun_final, \
                calparams_explore, ofun_explore

