import math
import logging
import numpy as np
import pandas as pd

from scipy.stats import t as student
from scipy.stats import shapiro

from scipy.optimize import fmin

import matplotlib.pyplot as plt

from hydrodiy.stat import sutils
import c_hydrodiy_stat


# Setup login
LOGGER = logging.getLogger(__name__)


def ar1_loglikelihood(theta, xmat, yvect, eps=1e-10):
    ''' Returns the components of the GLS ar1 log-likelihood '''

    sigma = theta[0]
    phi = theta[1]
    params = np.array(theta[2:]).reshape((len(theta)-2, 1))
    yvect_hat = np.dot(xmat, params)

    nval = yvect_hat.shape[0]
    resid = np.array(yvect-yvect_hat).reshape((nval, ))
    innov = sutils.ar1inverse([phi, 0.], resid)
    sse = np.sum(innov[1:]**2)
    sse += innov[0]**2 * (1-phi**2)

    #ll1 = -n/2*math.log(2*math.pi)
    if sigma>eps:
        loglike2 = -nval*math.log(sigma)
    else:
        loglike2 = -nval*math.log(eps) - (sigma-eps)**2*1e100

    if phi**2<1-eps:
        loglike3 = math.log(1-phi**2)/2
    else:
        loglike3 = math.log(1-eps**2)/2 - (phi**2-1+eps)**2*1e100

    loglike4 = -sse/(2*sigma**2)

    loglike = {'sigma':loglike2, 'phi':loglike3, 'sse':loglike4}

    return loglike



def ar1_loglikelihood_objfun(theta, xmat, yvect):
    ''' Returns the GLS ar1 log-likelihood objective function '''

    loglike = ar1_loglikelihood(theta, xmat, yvect)
    objfun = -(loglike['sigma']+loglike['phi']+loglike['sse'])

    return objfun



class Linreg:
    ''' Linear regression modelling '''

    def __init__(self, x, y,
            has_intercept=True,
            polyorder=1,
            type='ols',
            varnames=None):
        ''' Initialise regression model

            :param np.array x: Predictor variable
            :param np.array  y: Predictand variable
            :param bool has_intercept: Add intercept to regression model?
            :param int polyorder: Add polynomial terms with higher order (polyorder>1)
            :param str type: Regression type
            :param list varnames: Variable names

        '''

        # Store data
        self.type = type
        self.has_intercept = has_intercept
        self.polyorder = polyorder
        self.x = x
        self.y = y
        self.varnames = varnames
        self.nboot_print = 0

        # Build inputs
        self._buildinput()


    def __str__(self):
        str = '\n\t** Linear model **\n'
        str += '\n\tModel setup:\n'
        str += '\t  N predictors: %s\n' % self.npredictors
        str += '\t  N predictands: %s\n' % self.npredictands
        str += '\t  Type: %s\n'% self.type
        str += '\t  Has intercept: %s\n' % self.has_intercept
        str += '\t  Polynomial order: %d\n\n' % self.polyorder
        str += '\tParameters:\n'
        for idx, row in self.params.iterrows():

            if self.type == 'ols':
                str += '\t  %s = %5.2f [%5.2f, %5.2f] P(>|t|)=%0.3f\n'%(idx,
                    row['estimate'], row['2.5%'],
                    row['97.5%'], row['Pr(>|t|)'])

            if self.type == 'gls_ar1':
                str += '\t  %s = %5.2f\n' % (idx, row['estimate'])

        if self.type == 'gls_ar1':
            str += '\n\tAR1 coefficient (AR1 GLS only):\n'
            str += '\t  phi = %6.3f\n' % self.phi

            str += '\n\tLikelihood component (AR1 GLS only):\n'
            for k in self.loglikelihood:
                str += '\t  ll[%5s] = %6.3f\n' % (k, self.loglikelihood[k])

        str += '\n\tPerformance:\n'
        str += '\t  R2        = %6.3f\n' % self.diagnostic['R2']
        str += '\t  Bias      = %6.3f\n' % self.diagnostic['bias']
        str += '\t  Mean Error= %6.3f\n' % self.diagnostic['mean_error']
        str += '\t  Coef Det  = %6.3f\n' % self.diagnostic['coef_determination']
        str += '\t  Ratio Var = %6.3f\n' % self.diagnostic['ratio_variance']

        str += '\n\tTest on normality of residuals (Shapiro):\n'
        sh = self.diagnostic['shapiro_residuals_pvalue']
        mess = '(<0.05 : failing normality at 5% level)'
        str += '\t  P value = %6.3f %s\n' % (sh, mess)

        str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
        dw = self.diagnostic['durbinwatson_residuals_stat']
        mess = '(<1 : residuals may not be independent)'
        str += '\t  Statistic = %6.3f %s\n' % (dw, mess)

        return str


    def _buildXmatrix(self, xmat0=None):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.'''

        # Prediction data
        if xmat0 is None:
            use_internal = True
            xmat0 = np.array(self.x)
        else:
            use_internal = False
            xmat0 = np.array(xmat0)

        # Dimensions
        xmat0 = np.atleast_2d(xmat0).astype(np.float64)
        if xmat0.shape[0] == 1:
            xmat0 = xmat0.T
        nsamp = xmat0.shape[0]
        npred = xmat0.shape[1]

        if use_internal:
            self.npredictors = npred
        else:
            if npred != self.npredictors:
                raise ValueError(('Number of predictors in input data(%d)' + \
                    ' different from regression(%d)') %(npred, self.npredictors))

        # Produce X matrix
        xmat = np.empty((xmat0.shape[0], xmat0.shape[1] * self.polyorder),
                                np.float64)

        for j in range(xmat0.shape[1]):
            for k in range(self.polyorder):
                xmat[:, j*self.polyorder+k] = xmat0[:, j]**(k+1)

        if self.has_intercept:
            xmat = np.insert(xmat, 0, 1., axis=1)

        return xmat, nsamp, npred


    def _get_param_names(self):
        ''' Extract regression parameter names '''

        # Extract variable names
        if self.varnames is None:
            try:
                # Get varnames from pandas.DataFrame columns
                parnames = list(self.x.columns)

            except AttributeError:
                try:
                    # Get varnames from pandas.Series name
                    parnames = [self.x.name]

                except AttributeError:
                    parnames = ['x%2.2d' % k for k in range(self.npredictors)]
        else:
            parnames = self.varnames.copy()

        # Expand if polyorder>1
        if self.polyorder>1:
            pn = []
            for n in parnames:
                for k in range(self.polyorder):
                    pn.append('%s**%d' % (n, k+1))

            parnames = pn

        # Has intercept?
        if self.has_intercept:
            parnames.insert(0, 'intercept')

        return parnames


    def _buildinput(self):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.

        '''

        # Build X matrix
        xmat, nsamp, npredictors = self._buildXmatrix()
        self.xmat = xmat
        self.npredictors = npredictors
        self.nsample = nsamp

        # Get parameter names
        self.params_names = self._get_param_names()

        # build input matrix
        self.tXXinv = np.linalg.inv(np.dot(xmat.T,xmat))

        # build output matrix
        yvect = np.atleast_2d(np.array(self.y))
        if yvect.shape[0] == 1:
            yvect = yvect.T

        if yvect.shape[1] > 1:
            raise ValueError('More than one column in y data')

        if yvect.shape[0] != nsamp:
            import pdb; pdb.set_trace()
            raise ValueError(('Not the same number of samples in x({0})' + \
                ' and y({1})').format(nsamp, yvect.shape[0]))

        self.yvect = yvect.astype(np.float64)

        self.npredictands = self.yvect.shape[1]


    def _fit_ols(self):
        ''' Estimate parameter with ordinary least squares '''

        # OLS parameter estimate
        pars = np.dot(self.tXXinv, np.dot(self.xmat.T, self.yvect))
        ymat_hat = np.dot(self.xmat, pars)
        residuals = self.yvect-ymat_hat

        # Degrees of freedom (max 200 to avoid memory problems)
        degfree = self.yvect.shape[0]-len(pars)
        degfree = min(200, degfree)

        # Sigma of residuals and parameters
        sigma = math.sqrt(np.dot(residuals.T, residuals)/degfree)
        sigma_pars = sigma * np.sqrt(np.diag(self.tXXinv))

        # Parameter data frame
        params = pd.DataFrame({'estimate':pars[:,0], 'stderr':sigma_pars})
        params['parameter'] = self.params_names
        params = params.set_index('parameter')

        params['tvalue'] = params['estimate']/params['stderr']
        params['2.5%'] = params['estimate']+\
                            params['stderr']*student.ppf(2.5e-2, degfree)
        params['97.5%'] = params['estimate']+\
                            params['stderr']*student.ppf(97.5e-2, degfree)
        params['Pr(>|t|)'] = (1-student.cdf(np.abs(params['tvalue']), degfree))*2
        params = params[['estimate', 'stderr', '2.5%',
                            '97.5%', 'tvalue', 'Pr(>|t|)']]

        return params, sigma, degfree


    def _gls_transform_matrix(self, nsamp, phi):
        ''' Compute transformation matrix required for GLS iterative solution'''

        transmat = np.eye(nsamp)
        transmat[0,0] = math.sqrt(1-phi**2)
        transmat -= np.diag([1]*(nsamp-1), -1)*phi

        return transmat


    def _fit_gls_ar1(self):
        ''' Estimate parameter with generalized least squares
            assuming ar1 residuals
        '''

        # OLS regression to define starting point for optimisation
        params, sigma, _ = self._fit_ols()

        # Estimate auto-correlation of residuals
        pp = np.array(params['estimate']).reshape((params.shape[0],1))
        residuals = self.yvect-np.dot(self.xmat, pp)

        lm_residuals = Linreg(residuals[:-1], residuals[1:],
                            type='ols', has_intercept=False)
        lm_residuals.fit()
        phi = lm_residuals.params['estimate'].values[0]

        # Maximisation of log-likelihood
        theta0 = [sigma, phi] + list(params['estimate'])
        res = fmin(ar1_loglikelihood_objfun, theta0,
                    args=(self.xmat, self.yvect,), disp=0)

        # Build parameter dataframe
        params = pd.DataFrame({'estimate':res[2:]})
        params['parameter'] = self.params_names
        params = params.set_index('parameter')

        # Extract sigma and phi
        sigma = res[0]
        phi = res[1]

        if (phi<-1)|(phi>1):
            raise ValueError('Phi(%0.5f) is not within [-1, 1], Error in optimisation of log-likelihood' % phi)

        if sigma<0:
            raise ValueError(('Sigma({0}) is not positive, ' + \
                'Error in optimisation of log-likelihood').format(sigma))

        return params, phi, sigma


    def get_residuals(self, yvect, yvect_hat):
        ''' Generate model residuals '''

        # Compute residuals
        residuals = yvect-yvect_hat

        # Extract innovation from AR1 if GLS AR1
        if self.type == 'gls_ar1':
            r = residuals.reshape((len(residuals),))
            residuals = sutils.ar1inverse([self.phi, 0.], r)

        return residuals


    def compute_diagnostic(self, yvect, yvect_hat):
        ''' perform tests on regression assumptions '''

        residuals = self.get_residuals(yvect, yvect_hat)

        # Shapiro Wilks on residuals
        # resample if the number of residuals is greater than 5000
        nres = len(residuals)
        if nres < 5000:
            shap = shapiro(residuals)
        else:
            kk = np.random.choice(range(nres), 5000, replace=False)
            shap = shapiro(residuals[kk])

        # Durbin watson test
        residuals = residuals.reshape((len(residuals), ))
        differr = np.diff(residuals, 1)
        durbwat = np.dot(differr, differr)/np.dot(residuals, residuals)

        # correlation
        u = yvect-np.mean(yvect_hat)
        v = yvect_hat-np.mean(yvect_hat)
        rsquared = np.sum(u*v)**2/np.sum(u**2)/np.sum(v**2)

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
            'shapiro_residuals_stat': shap[0],
            'shapiro_residuals_pvalue':shap[1],
            'durbinwatson_residuals_stat': durbwat,
            'R2': rsquared
        }

        return diag


    def fit(self, log_entry=True):
        ''' Run parameter estimation and compute diagnostics '''

        # Fit
        if self.type == 'ols':
            params, sigma, degfree = self._fit_ols()
            phi = None

        elif self.type =='gls_ar1':
            params, phi, sigma = self._fit_gls_ar1()
            degfree = None

            pp = [sigma, phi] + list(params['estimate'])
            self.loglikelihood = ar1_loglikelihood(pp, self.xmat, self.yvect)

        else:
            raise ValueError('Regression type %s not recognised' % self.type)

        # Store data
        self.params = params
        self.sigma = sigma
        self.degfree = degfree
        self.phi = phi

        # compute fit
        yvect_hat = np.dot(self.xmat, self.params['estimate'])
        self.yvect_hat = yvect_hat.reshape((self.nsample, 1))

        # Run diagnostic
        diag = self.compute_diagnostic(self.yvect, self.yvect_hat)
        self.diagnostic = diag

        if log_entry:
            LOGGER.critical('Completed fit')
            LOGGER.critical(str(self))


    def predict(self, xmat=None, coverage=[95, 80]):
        ''' Pediction with confidence intervals

        Parameters
        -----------
        xmat : numpy.ndarray
            Regression inputs
        coverage : list
            Coverage probabilities

        Returns
        -----------
        yy0 : numpy.ndarray
            Regression outputs
        predint : pandas.DataFrame
            Prediction intervals
        '''

        # Generate regression inputs and output
        xmat, nsamp, npred = self._buildXmatrix(xmat)

        # Check dimensions
        params = self.params['estimate']
        if xmat.shape[1] != len(params):
            raise ValueError(('incorrect number of variables in x matrix ' + \
                '({0}), should be {1}').format(xmat.shape[1], len(params)))

        yvect_hat = np.dot(xmat, params)

        # Prediction intervals (only for OLS)
        predint = None

        if self.type == 'ols':

            # prediction factor
            # See Johnston, Econometric Methods, Equation 3.48
            predfact = np.zeros((nsamp,)).astype(np.float64)
            predfact = c_hydrodiy_stat.olspredfact(xmat, self.tXXinv,predfact)

            # Compute prediction intervals
            predint = {}
            for cov in coverage:
                proba = (100-cov)*5e-3
                stbounds = predfact*student.ppf(proba, self.degfree)

                c1 = '{0:0.1f}%'.format(100-100*proba)
                predint[c1] = yvect_hat-stbounds

                c2 = '{0:0.1f}%'.format(100*proba)
                predint[c2] = yvect_hat+stbounds

            predint = pd.DataFrame(predint)

        return yvect_hat, predint


    def boot(self, nsample=500):
        ''' Confidence intervals based on bootstrap '''

        # Performs first fit
        self.fit()
        residuals = self.get_residuals(self.yvect, self.yvect_hat)

        # Initialise boot data
        self.params_boot = []
        self.diagnostic_boot = []

        for i in range(nsample):

            if self.nboot_print > 0:
                if i%self.nboot_print==0:
                    mess = 'Boot sample {0:4d}/{1:4d}'.format(i+1, nsample)
                    LOGGER.critical(mess)

            # Resample residuals
            residuals_boot = residuals.copy().flatten()
            residuals_boot[1:] = np.random.choice(residuals[1:].flatten(), size=residuals.shape[0]-1)

            # Reconstruct autocorrelated signal in case of GLS_AR1
            # Resample only the 2, 3, ..., n values. The first value remains the same
            if self.type == 'gls_ar1':
                residuals_boot = sutils.ar1innov([self.phi, 0.], residuals_boot)

            # Create a new set of observations
            y_boot = self.yvect_hat.flatten() + residuals_boot

            # Fit regression
            lmboot = Linreg(self.x, y_boot,
                type=self.type,
                has_intercept=self.has_intercept,
                polyorder=self.polyorder,
                varnames=self.varnames)

            lmboot.fit(log_entry=False)

            # Store results
            self.params_boot.append(lmboot.params['estimate'].values)
            self.diagnostic_boot.append(lmboot.diagnostic)

        # Compute quantiles on bootstrap results
        self.params_boot = pd.DataFrame(self.params_boot,
            columns=lmboot.params.index,
            index=np.arange(nsample))

        perct = lambda x:  \
            sutils.percentiles(x, [2.5, 5, 10, 50, 90, 95, 97.5])

        self.params_boot_percentiles = self.params_boot.apply(perct).T

        self.diagnostic_boot = pd.DataFrame(self.diagnostic_boot,
            index=np.arange(nsample))

        self.diagnostic_boot_percentiles = self.diagnostic_boot.apply(perct).T

        LOGGER.critical('Completed bootstrap')


    def scatterplot(self, ax=None, log=False):
        ''' plot Yhat vs Y '''

        if ax is None:
            ax = plt.gca()

        if log:
            ax.loglog(self.yvect_hat, self.yvect, 'o')
        else:
            ax.plot(self.yvect_hat, self.yvect, 'o')

        # Set axes boundaries to get a square plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        # Show the 1:1 line
        ax.plot(lim, lim, 'k--')

        # Axis labels
        ax.set_xlabel(r'$\hat{Y}$')
        ax.set_ylabel(r'$Y$')

        # R2
        ax.annotate(r'$R^2$ = %0.2f' % self.diagnostic['R2'],
                va='top', ha='left',
                xy=(0.05, 0.95), xycoords='axes fraction')

