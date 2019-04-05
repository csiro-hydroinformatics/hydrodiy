import math
import logging
import numpy as np
import pandas as pd

from scipy.stats import t as student
from scipy.stats import shapiro

from scipy.optimize import fmin_powell as fmin

from hydrodiy.stat import sutils

# Try to import C code
HAS_C_STAT_MODULE = True
try:
    import c_hydrodiy_stat
except ImportError:
    HAS_C_STAT_MODULE = False

# Setup login
LOGGER = logging.getLogger(__name__)

# Tolerance of values of phi (phi**2 should be lower)
PHI_EPS = 1e-2

# Tolerance for values of sigma (sigma should be higher)
SIGMA_EPS = 1e-10

# Types of regressions
REGTYPES =['ols', 'gls_ar1']

# ------ Utility functions used in the Linreg class ----------

def data2vect(data):
    ''' Squeeze vectors to 1d  '''

    vect = np.atleast_1d(data)
    if vect.ndim == 2:
        vect = np.squeeze(vect)

    if vect.ndim == 0:
        vect = vect.reshape((1, ))

    if vect.ndim != 1:
        raise ValueError('Cannot convert to vector')

    return vect


def data2matrix(data):
    ''' convert data to 2D matrix '''

    mat = np.atleast_2d(data).astype(np.float64)

    if mat.ndim != 2:
        raise ValueError('Cannot convert to matrix')

    # Transpose 1d vector
    transpose_1d = True
    if hasattr(data, 'ndim'):
        if data.ndim == 2:
            transpose_1d = False

    if transpose_1d and mat.shape[0] == 1:
        mat = mat.T

    return mat


def get_xmat(xmat0, polyorder, intercept):
    ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
        or [xx] if the intercept is not included in the regression equation.'''

    # Dimensions
    xmat0 = data2matrix(xmat0)
    npredictors = xmat0.shape[1]

    # Produce X matrix
    xmat = np.empty((xmat0.shape[0], xmat0.shape[1] * polyorder),
                            np.float64)

    for j in range(xmat0.shape[1]):
        for k in range(polyorder):
            xmat[:, j*polyorder+k] = xmat0[:, j]**(k+1)

    if intercept:
        xmat = np.insert(xmat, 0, 1., axis=1)

    return xmat, npredictors


def get_names(data, polyorder, intercept, npredictors):
    ''' Extract regression parameter names '''

    # Extract variable names
    if hasattr(data, 'columns'):
        names = list(data.columns)
    else:
        names = ['x%2.2d' % k for k in range(npredictors)]

    # Expand if polyorder>1
    if polyorder>1:
        names_full = []
        for n in names:
            for k in range(polyorder):
                names_full.append('%s^%d' % (n, k+1))

        names = names_full

    # Has intercept?
    if intercept:
        names.insert(0, 'intercept')

    return names


def get_residuals(yvect, yvect_hat, regtype, phi):
    ''' Generate model residuals '''

    # Compute residuals for OLS
    yvect = data2vect(yvect)
    yvect_hat = data2vect(yvect_hat)
    residuals = yvect-yvect_hat

    if regtype == 'gls_ar1':
        if phi is None:
            raise ValueError('phi is none. ' + \
                'Cannot compute GLS AR1 residuals')

        # Extract innovation from AR1 if GLS AR1
        residuals = sutils.ar1inverse(phi, residuals, 0.)

    return residuals


# ------ Likelihood functions used to infer parameters ------------

def ar1_loglikelihood(theta, xmat, yvect):
    ''' Returns the components of the GLS ar1 log-likelihood '''

    sigma = theta[0]
    phi = theta[1]
    params = np.array(theta[2:]).reshape((len(theta)-2, 1))

    # Return 0 likelihood of parameters are outside bounds
    default = {'sigma':-np.inf, 'phi':-np.inf, 'sse':-np.inf}
    if abs(phi)>1-PHI_EPS:
        return default

    if sigma < SIGMA_EPS:
        return default

    # Compute predictions
    yvect_hat = np.dot(xmat, params).flat[:]

    nval = yvect_hat.shape[0]
    resid = np.array(yvect-yvect_hat).reshape((nval, ))
    innov = sutils.ar1inverse(phi, resid, 0.)
    sse = np.sum(innov[1:]**2)
    sse += innov[0]**2 * (1-phi**2)

    loglike = {
        'sigma':-nval*math.log(sigma),
        'phi':math.log(1-phi**2)/2,
        'sse':-sse/(2*sigma**2)
    }

    return loglike


def ar1_loglikelihood_objfun(theta, xmat, yvect):
    ''' Returns the GLS ar1 log-likelihood objective function '''

    loglike = ar1_loglikelihood(theta, xmat, yvect)
    objfun = -(loglike['sigma']+loglike['phi']+loglike['sse'])

    return objfun



# --------- Linear regression class ---------------

class Linreg:
    ''' Linear regression modelling '''

    def __init__(self, x, y,
            intercept=True,
            polyorder=1,
            regtype='ols'):
        '''
        Initialise a linear regression model

        Parameters
        -----------
        x : numpy.ndarray
            Predictors
        y : numpy.ndarray
            Predictand
        intercept : bool
            Does regression includes an intercept term?
        polyorder : int
            Order for polynomial regression. E.g. 2 leads to
            the regression y = a+bx+cx^2
        regtype : str
            Regression type. Available types are
                ols - Ordinary least squares
                gls_ar1 - Generalised least squares with
                        auto-regressive model for residuals
        '''

        # Store data
        self.intercept = intercept
        self.polyorder = polyorder
        self.nboot_print = 0

        if not regtype in REGTYPES:
            raise ValueError('regtype {0} not in {1}'.format(regtype, \
                '/'.join(REGTYPES)))

        self.regtype = regtype

        # Build inputs
        self.x = x
        self.xmat, npredictors = get_xmat(x, polyorder, intercept)
        self.names = get_names(x, polyorder, intercept,
                        npredictors)

        self.yvect = data2vect(y)

        # Check dimensions
        xdim = self.xmat.shape
        ydim = self.yvect.shape
        if xdim[0] != ydim[0]:
            raise ValueError('xdim[0]({0}) != ydim[0]({1})'.format(
                xdim[0], ydim[0]))

        if xdim[0] < xdim[1]:
            raise ValueError('Not enough data points ({0}) '+ \
                'for the number of parameters ({1})'.format(xdim[0],
                    xdim[1]))

        # Inference matrix
        self.tXXinv = np.linalg.inv(np.dot(self.xmat.T,self.xmat))


    def __str__(self):
        str = '\n\t** Linear model **\n'
        str += '\n\tModel setup:\n'
        str += '\t  Regression type: %s\n'% self.regtype
        str += '\t  Has intercept: %s\n' % self.intercept
        str += '\t  Polynomial order: %d\n\n' % self.polyorder

        if hasattr(self, 'params'):

            # compute boostrap confidence intervals
            if hasattr(self, 'params_boot'):
                qq = [2.5, 97.5]
                qqtxt = ['2.5%', '97.5%']
                boot_ci = self.params_boot.apply(lambda x:
                    pd.Series(np.percentile(x, qq), index=qqtxt))

            str += '\tParameters:\n'
            for idx, row in self.params.iterrows():

                if not hasattr(self, 'params_boot'):
                    if self.regtype == 'ols':
                        str += '\t  %9s = %5.3e [%5.3e, %5.3e] P(>|t|)=%0.3f\n'%(idx,
                            row['estimate'], row['2.5%'],
                            row['97.5%'], row['Pr(>|t|)'])

                    if self.regtype == 'gls_ar1':
                        str += '\t  %9s = %5.3e\n' % (idx, row['estimate'])

                else:
                    perct25 = boot_ci.loc['2.5%', idx]
                    perct975 = boot_ci.loc['97.5%', idx]
                    str += '\t  %9s = %5.3e [%5.3e, %5.3e] (boot)\n'%(idx,
                            row['estimate'], perct25, perct975)

            if self.regtype == 'gls_ar1':
                str += '\n\tAR1 coefficient (AR1 GLS only):\n'
                str += '\t  phi = %6.3f\n' % self.phi

                str += '\n\tLikelihood component (AR1 GLS only):\n'
                for k in self.loglikelihood:
                    str += '\t  ll[%5s] = %6.3f\n' % (k, self.loglikelihood[k])


        if hasattr(self, 'diagnostic'):
            str += '\n\tPerformance:\n'
            str += '\t  R2        = %6.3f\n' % self.diagnostic['R2']
            str += '\t  Bias      = %6.3f\n' % self.diagnostic['bias']
            str += '\t  Mean Error= %6.3f\n' % self.diagnostic['mean_error']
            str += '\t  Coef Det  = %6.3f\n' % self.diagnostic['coef_determination']
            str += '\t  Ratio Var = %6.3f\n' % self.diagnostic['ratio_variance']

            str += '\n\tTest on normality of residuals (Shapiro):\n'
            shap = self.diagnostic['shapiro_pvalue']
            mess = '(<0.05 : failing normality at 5% level)'
            str += '\t  P value = %6.3f %s\n' % (shap, mess)

            str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
            durbwat = self.diagnostic['durbinwatson_stat']
            mess = '(<1 : residuals may not be independent)'
            str += '\t  Statistic = %6.3f %s\n' % (durbwat, mess)

        return str


    def _fit_ols(self):
        ''' Estimate parameter with ordinary least squares '''

        # OLS parameter estimate
        ymat = data2matrix(self.yvect)
        pars = np.dot(self.tXXinv, np.dot(self.xmat.T, ymat))
        ymat_hat = np.dot(self.xmat, pars)
        residuals = ymat-ymat_hat

        pars = data2vect(pars)

        # Degrees of freedom (max 500 to avoid memory problems)
        degfree = self.yvect.shape[0]-len(pars)
        degfree_stud = degfree #min(500, degfree)

        # Sigma of residuals and parameters
        sigma = math.sqrt(np.dot(residuals.T, residuals)/degfree)
        sigma_pars = sigma * np.sqrt(np.diag(self.tXXinv))

        # Parameter data frame
        tvalue = pars/sigma_pars
        stud_25 = student.ppf(2.5e-2, degfree_stud)
        stud_975 = student.ppf(97.5e-2, degfree_stud)
        stud_cdf = (1-student.cdf(np.abs(tvalue), degfree_stud))*2

        params = pd.DataFrame({ \
                'estimate':pars, \
                'stderr':sigma_pars, \
                'tvalue':tvalue, \
                '2.5%':pars+sigma_pars*stud_25, \
                '97.5%':pars+sigma_pars*stud_975, \
                'Pr(>|t|)':stud_cdf \
            }, index=self.names)

        return params, sigma, degfree


    def _fit_gls_ar1(self):
        ''' Estimate parameter with generalized least squares
            assuming ar1 residuals
        '''

        # OLS regression to define starting point for optimisation
        params, sigma, _ = self._fit_ols()

        # Estimate auto-correlation of residuals
        pp = np.array(params['estimate']).reshape((params.shape[0],1))
        residuals = self.yvect-np.dot(self.xmat, pp).flat[:]
        phi = np.corrcoef(residuals[1:], residuals[:-1])[0, 1]

        if abs(phi) > 1-PHI_EPS:
            # Case where starting point is highly auto-correlated
            phi = (1-2*PHI_EPS)*np.sign(phi)

        # Maximisation of log-likelihood
        theta0 = [sigma, phi] + list(params['estimate'])
        res = fmin(ar1_loglikelihood_objfun, theta0,
                    args=(self.xmat, self.yvect,), disp=0)

        # Build parameter dataframe
        params = pd.DataFrame({'estimate':res[2:]})
        params['parameter'] = self.names
        params = params.set_index('parameter')

        # Extract sigma and phi
        sigma = max(SIGMA_EPS, res[0])
        phi = np.sign(res[1]) * min(1-PHI_EPS, abs(res[1]))

        if phi**2>1-PHI_EPS:
            raise ValueError(('Phi({0:0.5f}) is not within [-1, 1], Error in' + \
                ' optimisation of log-likelihood. Phi0 is {0:0.5f}').format( \
                phi, theta0[1]))

        if sigma<SIGMA_EPS:
            raise ValueError(('Sigma({0:0.5f}) is not strictly positive, ' + \
                'Error in optimisation of log-likelihood').format(sigma))

        return params, phi, sigma


    def compute_diagnostic(self, yvect, yvect_hat):
        ''' perform tests on regression assumptions

        Parameters
        -----------
        yvect : numpy.ndarray
            Observed data
        yvect_hat : numpy.ndarray
            Predicted data

        Returns
        -----------
        diag : dict
            Set of performance metrics
        '''

        residuals = get_residuals(yvect, yvect_hat,
                    self.regtype, self.phi)

        # Shapiro Wilks on residuals
        # resample if the number of residuals is greater than 5000
        nres = len(residuals)
        if nres < 5000:
            shap = shapiro(residuals)
        else:
            kk = np.random.choice(range(nres), 5000, replace=False)
            shap = shapiro(residuals[kk])

        # Durbin watson test
        differr = np.diff(residuals, 1)
        durbwat = np.dot(differr, differr)/np.dot(residuals, residuals)

        # correlation
        rsquared = np.corrcoef(yvect, yvect_hat)[0, 1]

        # Bias
        myvect = np.mean(yvect)
        merr = np.mean(yvect-yvect_hat)
        bias = merr/myvect

        # Coeff of determination
        coefdet = 1-np.sum((yvect-yvect_hat)**2)/np.sum((yvect-myvect)**2)

        # Ratio of variances
        ratio_std = np.std(yvect_hat)/np.std(yvect)

        # Store data
        diag = {
            'bias':bias,
            'mean_error':merr,
            'coef_determination':coefdet,
            'ratio_variance':ratio_std,
            'shapiro_stat': shap[0],
            'shapiro_pvalue':shap[1],
            'durbinwatson_stat': durbwat,
            'R2': rsquared
        }

        return diag


    def fit(self, use_logger=True, run_diagnostic=True):
        ''' Run parameter estimation and compute diagnostics

        Parameters
        -----------
        use_logger : bool
            Log entry to linreg.logger
        run_diagnostic : bool
            Compute diagnostic metrics
        '''

        # Fit
        if self.regtype == 'ols':
            params, sigma, degfree = self._fit_ols()
            phi = None

        elif self.regtype =='gls_ar1':
            params, phi, sigma = self._fit_gls_ar1()
            degfree = None

            pp = [sigma, phi] + list(params['estimate'])
            self.loglikelihood = ar1_loglikelihood(pp, self.xmat, self.yvect)

        # Store data
        self.params = params
        self.sigma = sigma
        self.degfree = degfree
        self.phi = phi

        # compute fit
        yvect_hat, _ = self.predict(coverage=None)

        # Run diagnostic
        if run_diagnostic:
            self.diagnostic = self.compute_diagnostic( \
                self.yvect, yvect_hat)

        if use_logger:
            LOGGER.info('Completed fit')
            LOGGER.info(str(self))


    def leverages(self):
        ''' Compute OLS regression leverages and cook distance
            See
            https://en.wikipedia.org/wiki/Leverage_(statistics)
            https://en.wikipedia.org/wiki/Studentized_residual
            https://en.wikipedia.org/wiki/Cook%27s_distance

        Returns
        -----------
        leverages : numpy.ndarray
            Leverage values for each regression point
        studentized_residuals : numpy.ndarray
            Studentized residuals
        cook : numpy.ndarray
            Cook's distance for each regression point
        '''
        if not HAS_C_STAT_MODULE:
            raise ValueError('C module c_hydrodiy_stat is not available, '+\
                'please run python setup.py build')

        if self.regtype != 'ols':
            raise ValueError('Can only compute leverage for ols regression')

        nsamp = self.xmat.shape[0]
        leverages = np.zeros((nsamp,)).astype(np.float64)
        c_hydrodiy_stat.olsleverage(self.xmat, self.tXXinv, leverages)

        yvect_hat, _ = self.predict(coverage=None)
        residuals = get_residuals(self.yvect, yvect_hat,
                self.regtype, self.phi)

        nval = len(residuals)
        nparams = self.params.shape[0]
        sse = np.sum(residuals**2)
        sigma_hat = sse/(nval-nparams)

        studentized_residuals = residuals/sigma_hat/np.sqrt(1-leverages)
        cook = residuals/sigma_hat*leverages/(1-leverages**2)

        return leverages, studentized_residuals, cook


    def predict(self, x=None, coverage=[95, 80], boot=False):
        ''' Pediction with confidence intervals

        Parameters
        -----------
        x : numpy.ndarray
            Regression inputs
        coverage : list
            Coverage probabilities. If None, does not compute prediction
            intervals
        boot : bool
            Use boostrap results

        Returns
        -----------
        yy0 : numpy.ndarray
            Regression outputs
        predint : pandas.DataFrame
            Prediction intervals
        '''
        if not HAS_C_STAT_MODULE:
            raise ValueError('C module c_hydrodiy_stat is not available, '+\
                'please run python setup.py build')

        # Generate regression inputs and output
        if x is None:
            x = self.x

        xmat, _ = get_xmat(x, self.polyorder,
                        self.intercept)

        # Check dimensions
        if not hasattr(self, 'params'):
            raise ValueError('No fitted params, please run fit')

        if xmat.shape[1] != self.params.shape[0]:
            raise ValueError('Expected x with {0} columns, got {1}'.format(\
                self.params.shape[0], xmat.shape[1]))

        # Compute prediction
        params = data2matrix(self.params['estimate'])
        yvect_hat = data2vect(np.dot(xmat, params))

        # Compute prediction intervals
        predint = None

        if coverage is None:
            return yvect_hat, predint

        if not boot:
            # Regular confidence intervales based on probability assumptions

            if self.regtype == 'ols':
                # prediction factor
                # See Johnston, Econometric Methods, Equation 3.48
                nsamp = xmat.shape[0]
                leverages = np.zeros((nsamp,)).astype(np.float64)
                c_hydrodiy_stat.olsleverage(xmat, self.tXXinv, leverages)
                predfact = np.sqrt(1+leverages) * self.sigma

                # Compute prediction intervals
                predint = {}
                for cov in coverage:
                    proba = (100-cov)*5e-3
                    stbounds = predfact*student.ppf(proba, self.degfree)

                    c1 = '{0:0.1f}%'.format(100-100*proba)
                    predint[c1] = yvect_hat-stbounds

                    c2 = '{0:0.1f}%'.format(100*proba)
                    predint[c2] = yvect_hat+stbounds

        else:
            if not hasattr(self, 'params_boot'):
                raise ValueError('No bootstrap results, '+\
                                    'please run Linreg.boot')

            # Computed predicted values with all bootstrap parameters
            pboot = self.params_boot
            yvect_boot = []
            for i in range(len(pboot)):
                params = data2matrix(pboot.iloc[i, :])
                yvect_boot.append(np.dot(xmat, params)[:, 0])

            yvect_boot = pd.DataFrame(np.array(yvect_boot).T)

            # Compute prediction intervals from bootstrap results
            predint = {}

            for cov in coverage:
                proba = (100.-cov)/2
                qq = [proba, 100.-proba]
                qqtxt = ['{0:0.1f}%'.format(qqq) for qqq in qq]

                tmp = yvect_boot.apply(lambda x: \
                        pd.Series(np.percentile(x, qq), qqtxt), axis=1)
                for qq in qqtxt:
                    predint[qq] = tmp[qq]

        # Reformat predint to dataframe
        if not predint is None:
            predint = pd.DataFrame(predint)

            # Reorder columns
            cc = [float(cn[:-1]) for cn in predint.columns]
            kk = np.argsort(cc)
            predint = predint.iloc[:, kk]

        return yvect_hat, predint


    def boot(self, nsample=500, run_diagnostic=False,
                nboot_print=10):
        ''' Bootstrap the regression fitting process

        Parameters
        -----------
        nsamples : int
            Number of bootstrap replicates
        run_diagnostic : bool
            Compute diagnostic metrics or not
        nboot_print : int
            Frequency of message logging during bootstrap
            process
        '''
        # Performs first fit
        self.fit()
        yvect_hat, _ = self.predict(coverage=None)
        residuals = get_residuals(self.yvect, yvect_hat,
                    self.regtype, self.phi)
        nresiduals = len(residuals)

        # Initialise boot data
        self.params_boot = []
        diagnostic_boot = []
        yvect_original = self.yvect.copy()

        for i in range(nsample):

            if self.nboot_print > 0:
                if i%self.nboot_print==0:
                    mess = 'Boot sample {0:4d}/{1:4d}'.format(i+1, nsample)
                    LOGGER.info(mess)

            # Resample residuals
            residuals_boot = residuals.copy()
            residuals_boot[1:] = np.random.choice(residuals[1:],
                                size=nresiduals-1)

            # Reconstruct autocorrelated signal in case of GLS_AR1
            # Resample only the 2, 3, ..., n values. The first value remains the same
            if self.regtype == 'gls_ar1':
                residuals_boot = sutils.ar1innov(self.phi, residuals_boot, 0.)

            # Create a new set of observations
            y_boot = yvect_hat + residuals_boot

            # Fit regression
            self.yvect = data2vect(y_boot)

            self.fit(use_logger=False, \
                run_diagnostic=run_diagnostic)

            # Store results
            self.params_boot.append(self.params['estimate'].values)

            if run_diagnostic:
                diagnostic_boot.append(self.diagnostic)

        # Restore observed data
        self.yvect = yvect_original

        # Compute quantiles on bootstrap results
        self.params_boot = pd.DataFrame(self.params_boot, \
            columns=self.params.index, \
            index=np.arange(nsample))

        if run_diagnostic:
            self.diagnostic_boot = pd.DataFrame(diagnostic_boot, \
                index=np.arange(nsample))

        LOGGER.info('Completed bootstrap')


