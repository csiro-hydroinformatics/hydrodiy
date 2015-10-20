import math

import numpy as np
import pandas as pd
from scipy.stats import t as student
from scipy.stats import shapiro

from scipy.optimize import fmin

import matplotlib.pyplot as plt

from hystat import sutils

def ar1_loglikelihood(theta, X, Y, eps=1e-10):
    ''' Returns the components of the GLS ar1 log-likelihood '''

    sigma = theta[0]
    phi = theta[1]
    p = np.array(theta[2:]).reshape((len(theta)-2,1))
    Yhat = np.dot(X,p)

    n = Yhat.shape[0]
    e = np.array(Y-Yhat).reshape((n,))
    innov = sutils.ar1inverse([phi, 0.], e)
    sse = np.sum(innov[1:]**2)
    sse += innov[0]**2 * (1-phi**2)

    #ll1 = -n/2*math.log(2*math.pi)
    if sigma>eps:
        ll2 = -n*math.log(sigma)
    else:
        ll2 = -n*math.log(eps) - (sigma-eps)**2*1e100

    if phi**2<1-eps:
        ll3 = math.log(1-phi**2)/2
    else:
        ll3 = math.log(1-eps**2)/2 - (phi**2-1+eps)**2*1e100

    ll4 = -sse/(2*sigma**2)

    ll = {'sigma':ll2, 'phi':ll3, 'sse':ll4}

    return ll

def ar1_loglikelihood_objfun(theta, X, Y):
    ''' Returns the GLS ar1 log-likelihood objective function '''

    ll = ar1_loglikelihood(theta, X, Y)
    lls = -(ll['sigma']+ll['phi']+ll['sse'])

    return lls

class Linreg:
    ''' Class to handle linear regression '''

    def __init__(self, x, y,
            has_intercept = True,
            polyorder = 1,
            type = 'ols',
            varnames = None):
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

        self.nboot_print = 50

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
                    row['estimate'], row['confint_025'],
                    row['confint_975'], row['Pr(>|t|)'])

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

    def _buildXmatrix(self, xx0=None):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.'''

        # Prediction data
        if xx0 is None:
            xx = np.array(self.x)
        else:
            xx = np.array(xx0)

        # Dimensions
        XX = np.atleast_2d(xx)
        if len(xx.shape) == 1:
            XX = XX.T
        nsamp = xx.shape[0]
        npred = XX.shape[1]

        if xx0 is None:
            self.npredictors = npred
        else:
            if npred != self.npredictors:
                raise ValueError(('Number of predictors in input data(%d)' + \
                    ' different from regression(%d)') %(npred, self.npredictors))

        # Produce X matrix
        X = np.empty((XX.shape[0], XX.shape[1] * self.polyorder), float)

        for j in range(XX.shape[1]):

            for k in range(self.polyorder):

                # populate X matrix
                X[:, j*self.polyorder+k] = XX[:, j]**(k+1)

        if self.has_intercept:
            X = np.insert(X, 0, 1., axis=1)

        return X, nsamp, npred

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
        X, nsamp, npredictors = self._buildXmatrix()
        self.X = X
        self.npredictors = npredictors
        self.nsample = nsamp

        # Get parameter names
        self.params_names = self._get_param_names()

        # build input and output matrix
        self.tXXinv = np.linalg.inv(np.dot(X.T,X))

        self.Y = np.atleast_2d(np.array(self.y)).reshape((nsamp, 1))
        self.npredictands = self.Y.shape[1]


    def _fit_ols(self):
        ''' Estimate parameter with ordinary least squares '''

        # OLS parameter estimate
        pars = np.dot(self.tXXinv, np.dot(self.X.T, self.Y))
        Yhat = np.dot(self.X, pars)
        residuals = self.Y-Yhat

        # Degrees of freedom
        df = self.Y.shape[0]-len(pars)

        # Sigma of residuals and parameters
        sigma = math.sqrt(np.dot(residuals.T, residuals)/df)
        sigma_pars = sigma * np.sqrt(np.diag(self.tXXinv))

        # Parameter data frame
        params = pd.DataFrame({'estimate':pars[:,0], 'stderr':sigma_pars})
        params['parameter'] = self.params_names
        params = params.set_index('parameter')

        params['tvalue'] = params['estimate']/params['stderr']
        params['confint_025'] = params['estimate']+\
                            params['stderr']*student.ppf(2.5e-2, df)
        params['confint_975'] = params['estimate']+\
                            params['stderr']*student.ppf(97.5e-2, df)
        params['Pr(>|t|)'] = (1-student.cdf(np.abs(params['tvalue']), df))*2
        params = params[['estimate', 'stderr', 'confint_025',
                            'confint_975', 'tvalue', 'Pr(>|t|)']]

        return params, sigma, df

    def _gls_transform_matrix(self, nsamp, phi):
        ''' Compute transformation matrix required for GLS iterative solution'''

        P = np.eye(nsamp)
        P[0,0] = math.sqrt(1-phi**2)
        P -= np.diag([1]*(nsamp-1), -1)*phi

        return P

    def _fit_gls_ar1(self):
        ''' Estimate parameter with generalized least squares
            assuming ar1 residuals
        '''

        # OLS regression to define starting point for optimisation
        params, sigma, df = self._fit_ols()

        # Estimate auto-correlation of residuals
        pp = np.array(params['estimate']).reshape((params.shape[0],1))
        residuals = self.Y-np.dot(self.X, pp)
        lm_residuals = Linreg(residuals[:-1], residuals[1:], type='ols', has_intercept=False)
        lm_residuals.fit()
        phi = lm_residuals.params['estimate'].values[0]

        # Maximisation of log-likelihood
        theta0 = [sigma, phi] + list(params['estimate'])
        res = fmin(ar1_loglikelihood_objfun, theta0, args=(self.X, self.Y,), disp=0)

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
            import pdb; pdb.set_trace()
            raise ValueError('Sigma(%0.5f) is not within [0, +inf], Error in optimisation of log-likelihood' % sigma)

        return params, phi, sigma

    def getresiduals(self, Y, Yhat):

        # Compute residuals
        residuals = Y-Yhat

        # Extract innovation from AR1 if GLS AR1
        if self.type == 'gls_ar1':
            r = residuals.reshape((len(residuals),))
            residuals = sutils.ar1inverse([self.phi, 0.], r)

        return residuals

    def compute_diagnostic(self, Y, Yhat):
        ''' perform tests on regression assumptions '''

        residuals = self.getresiduals(Y, Yhat)

        # Shapiro Wilks on residuals
        s = shapiro(residuals)

        # Durbin watson test
        residuals = residuals.reshape((len(residuals), ))
        de = np.diff(residuals, 1)
        dw = np.dot(de, de)/np.dot(residuals, residuals)

        # correlation
        u = Y-np.mean(Y)
        v = Yhat-np.mean(Yhat)
        R2 = np.sum(u*v)**2/np.sum(u**2)/np.sum(v**2)

        # Bias
        mY = np.mean(Y)
        me = np.mean(Y-Yhat)
        b = me/mY

        # Coeff of determination
        d = 1-np.sum((Y-Yhat)**2)/np.sum((Y-mY)**2)

        # Ratio of variances
        rv = np.var(Yhat)/np.var(Y)

        # Store data
        diag = {'bias':b,
            'mean_error':me,
            'coef_determination':d,
            'ratio_variance':rv,
            'shapiro_residuals_stat': s[0],
            'shapiro_residuals_pvalue':s[1],
            'durbinwatson_residuals_stat': dw,
            'R2': R2}

        return diag


    def fit(self):
        ''' Run parameter estimation and compute diagnostics '''

        # Fit
        if self.type == 'ols':
            params, sigma, df = self._fit_ols()
            phi = None

        elif self.type =='gls_ar1':
            params, phi, sigma = self._fit_gls_ar1()
            df = None

            pp = [sigma, phi] + list(params['estimate'])
            self.loglikelihood = ar1_loglikelihood(pp, self.X, self.Y)

        else:
            raise ValueError('Regression type %s not recognised' % self.type)

        # Store data
        self.params = params
        self.sigma = sigma
        self.df = df
        self.phi = phi

        # compute fit
        Yhat = np.dot(self.X, self.params['estimate'])
        self.Yhat = Yhat.reshape((self.nsample, 1))

        # Run diagnostic
        diag = self.compute_diagnostic(self.Y, self.Yhat)
        self.diagnostic = diag

    def predict(self, x0=None, coverage=[95, 80]):
        ''' Pediction with intervals

            :param numpy.array x0: regression input
            :param list quantiles: Coverage of prediction intervals
        '''

        # Generate regression inputs
        X0, nsamp, npred = self._buildXmatrix(x0)

        Y0 = np.dot(X0, self.params['estimate'])
        nq = len(coverage)*2

        # Prediction intervals (only for OLS)
        PI = None

        if self.type == 'ols':
            PI = pd.DataFrame(np.zeros((nsamp ,nq)))
            cols = []
            for c in coverage:
                cols += ['predint_%3.3d'%(5*(100.-c)),
                            'predint_%3.3d'%(1000-5*(100.-c))]
            PI.columns = cols

            # prediction factor
            v = np.dot(X0, self.tXXinv)
            pf = self.sigma * np.sqrt(1.+np.dot(v, X0.T))

            # Compute prediction intervals
            for c in coverage:
                q = (100.-c)*5
                st = pf*student.ppf(q*1e-3, self.df)

                c1 = 'predint_%3.3d'%q
                PI[c1] = Y0+np.diag(st)

                c2 = 'predint_%3.3d'%(1000-q)
                PI[c2] = Y0-np.diag(st)

        return Y0, PI

    def boot(self, nsample=500):
        ''' Confidence intervals based on bootstrap '''

        # Performs first fit
        self.fit()
        residuals = self.getresiduals(self.Y, self.Yhat)

        # Initialise boot data
        self.params_boot = []
        self.diagnostic_boot = []

        for i in range(nsample):
            if i%self.nboot_print==0:
                print('\t\t.. Boot sample %4d / %4d ..' % (i+1, nsample))

            # Resample residuals
            residuals_boot = residuals.copy().flatten()
            residuals_boot[1:] = np.random.choice(residuals[1:].flatten(), size=residuals.shape[0]-1)

            # Reconstruct autocorrelated signal in case of GLS_AR1
            # Resample only the 2, 3, ..., n values. The first value remains the same
            if self.type == 'gls_ar1':
                residuals_boot = sutils.ar1innov([self.phi, 0.], residuals_boot)

            # Create a new set of observations
            y_boot = self.Yhat.flatten() + residuals_boot

            # Fit regression
            lmboot = Linreg(self.x, y_boot,
                type=self.type,
                has_intercept=self.has_intercept,
                polyorder=self.polyorder,
                varnames=self.varnames)

            lmboot.fit()

            # Store results
            self.params_boot.append(lmboot.params['estimate'])
            self.diagnostic_boot.append(lmboot.diagnostic)

        # Compute quantiles on bootstrap results
        self.params_boot = pd.DataFrame(self.params_boot)
        fp = lambda x: sutils.percentiles(x, [2.5, 5, 10, 50, 90, 95, 97.5])
        self.params_boot_percentiles = self.params_boot.apply(fp).T

        self.diagnostic_boot = pd.DataFrame(self.diagnostic_boot)
        self.diagnostic_boot_percentiles = self.diagnostic_boot.apply(fp).T

    def scatterplot(self, ax=None, log=False):
        ''' plot Yhat vs Y '''

        if ax is None:
            ax = plt.gca()

        if log:
            ax.loglog(self.Yhat, self.Y, 'o')
        else:
            ax.plot(self.Yhat, self.Y, 'o')

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
                xy=(0.05, 0.93), xycoords='axes fraction')

