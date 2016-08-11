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


# ------ Utility functions used in the Linreg class ----------

def squeeze(vect):
    ''' Squeeze vectors to 1d  '''
    vects = np.squeeze(vect)

    if vects.ndim == 0:
        vects = vects.reshape((1, ))

    return vects

def data2matrix(data):
    ''' convert data to 2D matrix '''

    mat = np.atleast_2d(data).astype(np.float64)

    # Transpose 1d vector
    transpose_1d = True
    if hasattr(data, 'ndim'):
        if data.ndim == 2:
            transpose_1d = False

    if transpose_1d and mat.shape[0] == 1:
        mat = mat.T

    return mat


def get_xmat(xmat0, polyorder, has_intercept):
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

    if has_intercept:
        xmat = np.insert(xmat, 0, 1., axis=1)

    return xmat, npredictors


def get_names(data, polyorder, has_intercept, npredictors):
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
                names_full.append('%s**%d' % (n, k+1))

        names = names_full

    # Has intercept?
    if has_intercept:
        names.insert(0, 'intercept')

    return names


def get_residuals(yvect, yvect_hat, regtype, phi):
    ''' Generate model residuals '''

    # Compute residuals for OLS
    yvect = squeeze(data2matrix(yvect))
    yvect_hat = squeeze(data2matrix(yvect_hat))
    residuals = yvect-yvect_hat

    if regtype == 'gls_ar1':
        if phi is None:
            raise ValueError('phi is none. ' + \
                'Cannot compute GLS AR1 residuals')

        # Extract innovation from AR1 if GLS AR1
        residuals = sutils.ar1inverse([phi, 0.], residuals)

    return residuals


# ------ Likelihood functions used to infer parameters ------------

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



# --------- Linear regression class ---------------

class Linreg:
    ''' Linear regression modelling '''

    def __init__(self, x, y,
            has_intercept=True,
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
        has_intercept : bool
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
        self.has_intercept = has_intercept
        self.polyorder = polyorder
        self.nboot_print = 0

        if not regtype in ['ols', 'gls_ar1']:
            raise ValueError('regtype {0} not recognised'.format(regtype))

        self.regtype = regtype

        # Build inputs
        self.x = x
        self.xmat, npredictors = get_xmat(x, polyorder, has_intercept)
        self.names = get_names(x, polyorder, has_intercept,
                        npredictors)
        self.y = y
        self.yvect = data2matrix(y)

        # Check dimensions
        xdim = self.xmat.shape
        ydim = self.yvect.shape
        if xdim[0] != ydim[0]:
            raise ValueError('xdim[0]({0}) != ydim[0]({1})'.format(
                xdim[0], ydim[0]))

        if ydim[1] != 1:
            raise ValueError('ydim[0]({1}) != 1'.format(
                ydim[1]))

        # Inference matrix
        self.tXXinv = np.linalg.inv(np.dot(self.xmat.T,self.xmat))


    def __str__(self):
        str = '\n\t** Linear model **\n'
        str += '\n\tModel setup:\n'
        str += '\t  Type: %s\n'% self.regtype
        str += '\t  Has intercept: %s\n' % self.has_intercept
        str += '\t  Polynomial order: %d\n\n' % self.polyorder

        if hasattr(self, 'params'):

            # compute boostrap confidence intervals
            if hasattr(self, 'params_boot'):
                boot_ci = self.params_boot.apply(lambda x:
                    sutils.percentiles(x, [2.5, 97.5]))

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
            shap = self.diagnostic['shapiro_residuals_pvalue']
            mess = '(<0.05 : failing normality at 5% level)'
            str += '\t  P value = %6.3f %s\n' % (shap, mess)

            str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
            durbwat = self.diagnostic['durbinwatson_residuals_stat']
            mess = '(<1 : residuals may not be independent)'
            str += '\t  Statistic = %6.3f %s\n' % (durbwat, mess)

        return str


    def _fit_ols(self):
        ''' Estimate parameter with ordinary least squares '''

        # OLS parameter estimate
        pars = np.dot(self.tXXinv, np.dot(self.xmat.T, self.yvect))
        ymat_hat = np.dot(self.xmat, pars)
        residuals = self.yvect-ymat_hat

        # Degrees of freedom (max 200 to avoid memory problems)
        degfree = self.yvect.shape[0]-len(pars)
        degfree_stud= min(200, degfree)

        # Sigma of residuals and parameters
        sigma = math.sqrt(np.dot(residuals.T, residuals)/degfree)
        sigma_pars = sigma * np.sqrt(np.diag(self.tXXinv))

        # Parameter data frame
        params = pd.DataFrame({'estimate':pars[:,0], 'stderr':sigma_pars})
        params['parameter'] = self.names

        params = params.set_index('parameter')

        params['tvalue'] = params['estimate']/params['stderr']

        params['2.5%'] = params['estimate']+\
                            params['stderr']*student.ppf(2.5e-2, degfree_stud)

        params['97.5%'] = params['estimate']+\
                            params['stderr']*student.ppf(97.5e-2, degfree_stud)

        params['Pr(>|t|)'] = (1-student.cdf(np.abs(params['tvalue']),
                                        degfree_stud))*2
        params = params[['estimate', 'stderr', '2.5%',
                            '97.5%', 'tvalue', 'Pr(>|t|)']]

        return params, sigma, degfree


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
                            regtype='ols', has_intercept=False)
        lm_residuals.fit(False)
        phi = lm_residuals.params['estimate'].values[0]

        # Maximisation of log-likelihood
        theta0 = [sigma, phi] + list(params['estimate'])
        res = fmin(ar1_loglikelihood_objfun, theta0,
                    args=(self.xmat, self.yvect,), disp=0)

        # Build parameter dataframe
        params = pd.DataFrame({'estimate':res[2:]})
        params['parameter'] = self.names
        params = params.set_index('parameter')

        # Extract sigma and phi
        sigma = res[0]
        phi = res[1]

        if (phi<-1+1e-5)|(phi>1-1e-5):
            raise ValueError(('Phi({0:0.3f}) is not within [-1, 1], Error in' + \
                ' optimisation of log-likelihood').format(phi))

        if sigma<0:
            raise ValueError(('Sigma({0}) is not positive, ' + \
                'Error in optimisation of log-likelihood').format(sigma))

        return params, phi, sigma


    def compute_diagnostic(self, yvect, yvect_hat):
        ''' perform tests on regression assumptions '''

        residuals = get_residuals(yvect, yvect_hat,
                    self.regtype, self.phi)

        # Shapiro Wilks on residuals
        # resample if the number of residuals is greater than 5000
        residuals = squeeze(residuals)
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
            'shapiro_residuals_stat': shap[0],
            'shapiro_residuals_pvalue':shap[1],
            'durbinwatson_residuals_stat': durbwat,
            'R2': rsquared
        }

        return diag


    def fit(self, log_entry=True, run_diagnostic=True):
        ''' Run parameter estimation and compute diagnostics '''

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
                squeeze(self.yvect), yvect_hat)

        if log_entry:
            LOGGER.critical('Completed fit')
            LOGGER.critical(str(self))


    def leverages(self):
        ''' Compute OLS regression leverages and cook distance
            See

        Returns
        -----------
        leverages : numpy.ndarray
            Leverage values for each regression point
        cook : numpy.ndarray
            Cook's distance for each regression point
        '''

        if self.regtype != 'ols':
            raise ValueError('Can only compute leverage for ols regression')

        nsamp = self.xmat.shape[0]
        leverages = np.zeros((nsamp,)).astype(np.float64)
        c_hydrodiy_stat.olsleverage(self.xmat, self.tXXinv, leverages)

        yvect_hat, _ = self.predict(coverage=None)
        residuals = get_residuals(self.yvect, yvect_hat,
                self.regtype, self.phi)

        sse = np.sum(residuals**2)
        nparams = self.params.shape[0]
        nval = len(residuals)
        fact = float(nval-nparams)/nval
        cook = residuals/sse*leverages/(1-leverages**2)*fact

        return leverages, cook


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

        # Generate regression inputs and output
        if x is None:
            x = self.x

        xmat, _ = get_xmat(x, self.polyorder,
                        self.has_intercept)

        # Check dimensions
        if not hasattr(self, 'params'):
            raise ValueError('No fitted params, please run fit')

        if xmat.shape[1] != self.params.shape[0]:
            raise ValueError('x dimensons is inconsistent with params')

        # Compute prediction
        params = data2matrix(self.params['estimate'])

        yvect_hat = squeeze(np.dot(xmat, params))

        # Compute prediction intervals
        predint = None

        if coverage is None:
            return yvect_hat, predint

        if not boot:

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
                raise ValueError('No bootstrap results, please run boot')

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

                c1 = '{0:0.1f}%'.format(proba)
                predint[c1] = yvect_boot.apply(sutils.percentiles,
                    args=([proba], ), axis=1).squeeze()

                c2 = '{0:0.1f}%'.format(100-proba)
                predint[c2] = yvect_boot.apply(sutils.percentiles,
                    args=([100.-proba], ), axis=1).squeeze()

        # Reformat predint to dataframe
        if not predint is None:
            predint = pd.DataFrame(predint)

            # Reorder columns
            cc = [float(cn[:-1]) for cn in predint.columns]
            kk = np.argsort(cc)
            predint = predint.iloc[:, kk]


        return yvect_hat, predint


    def boot(self, nsample=500, run_diagnostic=False):
        ''' Bootstrap the regression fitting process '''

        # Performs first fit
        self.fit()
        yvect_hat, _ = self.predict(coverage=None)
        residuals = get_residuals(self.yvect, yvect_hat,
                    self.regtype, self.phi)
        nresiduals = len(residuals)

        # Initialise boot data
        self.params_boot = []
        diagnostic_boot = []

        for i in range(nsample):

            if self.nboot_print > 0:
                if i%self.nboot_print==0:
                    mess = 'Boot sample {0:4d}/{1:4d}'.format(i+1, nsample)
                    LOGGER.critical(mess)

            # Resample residuals
            residuals_boot = residuals.copy()
            residuals_boot[1:] = np.random.choice(residuals[1:],
                                size=nresiduals-1)

            # Reconstruct autocorrelated signal in case of GLS_AR1
            # Resample only the 2, 3, ..., n values. The first value remains the same
            if self.regtype == 'gls_ar1':
                residuals_boot = sutils.ar1innov([self.phi, 0.], residuals_boot)

            # Create a new set of observations
            y_boot = yvect_hat + residuals_boot

            # Fit regression
            self.yvect = data2matrix(y_boot)

            self.fit(log_entry=False, \
                run_diagnostic=run_diagnostic)

            # Store results
            self.params_boot.append(self.params['estimate'].values)

            if run_diagnostic:
                diagnostic_boot.append(self.diagnostic)

        # Restore observed data
        self.yvect = data2matrix(self.y)

        # Compute quantiles on bootstrap results
        self.params_boot = pd.DataFrame(self.params_boot, \
            columns=self.params.index, \
            index=np.arange(nsample))

        if run_diagnostic:
            self.diagnostic_boot = pd.DataFrame(diagnostic_boot, \
                index=np.arange(nsample))

        LOGGER.critical('Completed bootstrap')


    def scatterplot(self, ax=None,
        y=None, x=None,
        boot=False,
        coverage=95,
        set_square_bounds=False):
        '''
        Scattre plot of predicted versus observed dta with
        confidence prediction intervals

        Parameters
        -----------
        ax :
            Matplotlib axe to draw the plot
        x : numpy.ndarray
            Predictors
        y : numpy.ndarray
            Observed data
        boot : bool
            Use bootstrap  condifence intervals
        coverage : float
            Prediction intervals coverage (>0 and <100)
            Do not show intervals if coverage is None
        set_square_bounds : bool
            Set same bounds for x and y axis
        '''

        if not hasattr(self, 'params'):
            raise ValueError('No params, please run fit')

        if (x is None and not y is None) or \
            (not x is None and y is None):
            raise ValueError('Both x and y should be None or not None')

        # Build input data
        if y is None:
            yvect = self.yvect
        else:
            yvect = data2matrix(y)

        yvect = squeeze(yvect)

        if x is None:
            x = self.x

        # Generate prediction data
        yvect_hat, predint = self.predict(x=x, boot=boot,
                                coverage=[coverage])

        # Check dimensions
        if len(yvect_hat) != len(yvect):
            raise ValueError('Inconsistent length between' + \
                ' yvect({0}) and yvect_hat({1})'.format( \
                    len(yvect), len(yvect_hat)))

        # Grab current axe is none provided
        if ax is None:
            ax = plt.gca()

        # sort data
        kk = np.argsort(yvect_hat)

        # Draw scatter
        color = '#1f77b4'
        ax.plot(yvect_hat[kk], yvect[kk], 'o', \
            mfc=color, mec=color, alpha=0.9,
            label='fit')

        # Set axes boundaries to get a square plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]

        # Draw predicton intervals
        if not predint is None:
            color = '#1f77b4'
            label = '{0}% confidence intervals'.format(coverage)
            ax.plot(yvect_hat[kk], predint.iloc[kk, 0], '-', \
                color=color,label=label)

            ax.plot(yvect_hat[kk], predint.iloc[kk, 1], '-', \
                color=color)

        # Show the 1:1 line
        ax.plot(lim, lim, '-', color='grey')

        # Show high leverage points
        if self.regtype == 'ols' and y is None:
            leverages, cook = self.leverages()
            idx =  np.abs(cook[kk])>1
            ncook = np.sum(idx)
            color = '#d62728'
            ax.plot(yvect_hat[kk][idx], yvect[kk][idx], 'o', \
                mfc=color, mec=color, alpha=0.9,
                label='Cook D>1 ({0} points)'.format(ncook))

        # Set axe bounds
        if set_square_bounds:
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Axis labels
        ax.set_xlabel(r'$\hat{Y}$')
        ax.set_ylabel(r'$Y$')

        # legend
        leg = ax.legend(loc=4, numpoints=1,
            fancybox=True, fontsize='small')
        leg.get_frame().set_alpha(0.7)

        # R2
        if hasattr(self, 'diagnostic'):
            ax.annotate(r'$R^2$ = %0.2f' % self.diagnostic['R2'],
                va='top', ha='left',
                xy=(0.05, 0.95), xycoords='axes fraction')

