import math

import numpy as np
import pandas as pd
from scipy.stats import t as student
from scipy.stats import shapiro

from scipy.optimize import fmin

import matplotlib.pyplot as plt

from hystat import sutils

def ar1_loglikelihood(theta, X, Y):
    ''' Returns the opposite (x -1) of the GLS ar1 log-likelihood '''

    sigma = theta[0]
    phi = theta[1]
    p = np.array(theta[2:]).reshape((len(theta)-2,1))
    Yhat = np.dot(X,p)

    n = Yhat.shape[0]
    e = np.array(Y-Yhat).reshape((n,))
    innov = sutils.ar1inverse([phi, 0.], e)

    sse = np.sum(innov**2)
    ll = -n/2*math.log(math.pi)-n*math.log(sigma)+math.log(1-phi**2)/2-sse/(2*sigma)

    return -ll


class Linreg:
    ''' Class to handle linear regression '''

    def __init__(self, x, y, has_intercept = True, polyorder = 1, 
            type = 'ols', 
            gls_nexplore = 50, 
            gls_niterations = 20,
            gls_epsilon = 1e-4):
        ''' Initialise regression model

            :param np.array x: Predictor variable
            :param np.array  y: Predictand variable
            :param bool has_intercept: Add intercept to regression model?
            :param int polyorder: Add polynomial terms with higher order (polyorder>1)
            :param str type: Regression type
            :param int gls_nexplore: Number of initial AR1 values tested
            :param int gls_niterations: Number of iterations in GLS procedure
            :param int gls_epsilon: Convergence threshold for ar1 in gls iterative procedure

        '''

        # Store data
        self.type = type
        self.has_intercept = has_intercept
        self.polyorder = polyorder
        self.x = x
        self.y = y

        # GLS parameters
        self.gls_nexplore = gls_nexplore
        self.gls_niterations = gls_niterations
        self.gls_epsilon = gls_epsilon

        # Build inputs
        self._buildinput()

    def __str__(self):
        str = '\n\t** Linear model **\n'
        str += '\n\tModel setup:\n'
        str += '\t  N predictors: %s\n'%self.npredictors
        str += '\t  N predictands: %s\n'%self.npredictands
        str += '\t  Type: %s\n'%self.type
        str += '\t  Has intercept: %s\n'%self.has_intercept
        str += '\t  Polynomial order: %d\n\n'%self.polyorder
        str += '\tParameters:\n'
        for idx, row in self.params.iterrows():
            str += '\t  params %d = %5.2f [%5.2f, %5.2f] P(>|t|)=%0.3f\n'%(idx, 
                row['estimate'], row['confint_025'], 
                row['confint_975'], row['Pr(>|t|)'])

        if self.type == 'gls_ar1':
            str += '\n\tAR1 coefficient (AR1 GLS only):\n'
            str += '\t  phi = %0.3f\n'%self.phi
 
        str += '\n\tPerformance:\n'
        str += '\t  R2        = %0.3f\n'%self.diag['R2']
        str += '\t  Bias      = %0.3f\n'%self.diag['bias']
        str += '\t  Coef Det  = %0.3f\n'%self.diag['coef_determination']
        str += '\t  Ratio Var = %0.3f\n'%self.diag['ratio_variance']

        str += '\n\tTest on normality of residuals (Shapiro):\n'
        sh = self.diag['shapiro_pvalue']
        mess = '(<0.05 : failing normality at 5% level)'
        str += '\t  P value = %0.3f %s\n'%(sh, mess)

        str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
        dw = self.diag['durbinwatson_stat']
        mess = '(<1 : residuals may not be independent)'
        str += '\t  Statistic = %0.3f %s\n'%(dw, mess)
        
        return str

    def _getDim(self, xx):

        xx = np.array(xx)

         # Dimensions
        nsamp = xx.shape[0]
        
        if len(xx.shape) == 1:
            npred = 1
        else:
            npred = xx.shape[-1]

        return nsamp, npred

    def _buildXmatrix(self, xx0=None):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.'''

        # Prediction data
        if xx0 is None:
            xx = np.array(self.x)
        else:
            xx = np.array(xx0)

        # Dimensions
        nsamp, npred = self._getDim(xx)

        if xx0 is None:
            self.npredictors = npred
        else:
            if npred != self.npredictors:
                raise ValueError('Number of predictors in input data(%d) different from regression(%d)' %(npred,
                    self.npredictors))

        # Reshape array
        XX = xx.reshape(nsamp, self.npredictors)

        # Produce X matrix
        X = np.empty((XX.shape[0], XX.shape[1] * self.polyorder), float)
        for j in range(XX.shape[1]):
            for k in range(self.polyorder):
                X[:, j*self.polyorder+k] = XX[:, j]**(k+1)

        if self.has_intercept:
            X = np.insert(X, 0, 1., axis=1)

        return X, nsamp, npred


    def _buildinput(self):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.

        '''
 
        # Build X matrix
        X, nsamp, npredictors = self._buildXmatrix()
        self.X = X
        self.npredictors = npredictors
        self.nsample = nsamp

        # build input and output matrix
        self.tXXinv = np.linalg.inv(np.dot(X.T,X))

        self.Y = np.array(self.y).reshape((nsamp, 1))
        self.npredictands = self.Y.shape[1]

    def _ols(self):
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
        
        params['tvalue'] = params['estimate']/params['stderr']
        params['confint_025'] = params['estimate']+\
                            params['stderr']*student.ppf(2.5e-2, df)
        params['confint_975'] = params['estimate']+\
                            params['stderr']*student.ppf(97.5e-2, df)
        params['Pr(>|t|)'] = (1-student.cdf(np.abs(params['tvalue']), df))*2
        params = params[['estimate', 'stderr', 'confint_025', 
                            'confint_975', 'tvalue', 'Pr(>|t|)']]

        return params, sigma, df

    def _gls_transform_matrix(self, nsamp, ac1):
        ''' Compute transformation matrix required for GLS iterative solution'''

        P = np.eye(nsamp)
        P[0,0] = math.sqrt(1-ac1**2)
        P -= np.diag([1]*(nsamp-1), -1)*ac1

        return P

    def _gls_ar1(self):
        ''' Estimate parameter with generalized least squares 
            assuming ar1 residuals
        '''
        
        # OLS regression to define starting point for optimisation
        params, sigma, df = self._ols()

        # Estimate auto-correlation of residuals
        pp = np.array(params['estimate']).reshape((params.shape[0],1))
        residuals = self.Y-np.dot(self.X, pp)
        lm_residuals = Linreg(residuals[:-1], residuals[1:], type='ols', has_intercept=False)
        lm_residuals.fit()
        phi = lm_residuals.params['estimate'].values[0]

        # Maximisation of log-likelihood
        theta0 = [sigma, phi] + list(params['estimate'])
        res = fmin(ar1_loglikelihood, theta0, args=(self.X, self.Y,))
        params = pd.DataFrame({'estimate':res[2:]})
        sigma = res[0]
        phi = res[1]

        #iter = self.gls_niterations
        #pts = X.shape[0]
        #arams_gls_iter = np.empty((X.shape[1]+1, niter))

        # Systematic exploration of initial AR1 values
        #in_sigma = np.inf
        #c1_optim = 0.
        #
        #or ac1 in np.linspace(-0.99, 0.99, self.gls_nexplore):
        #   P = self._gls_transform_matrix(nsamp, ac1)
        #   Xs = np.dot(P, X)
        #   Ys = np.dot(P, Y)
        #   tXXinvs = np.linalg.inv(np.dot(Xs.T,Xs))
        #   params, sigma, df = self._ols(tXXinvs, Xs, Ys)

        #   if sigma < min_sigma:
        #       min_sigma = sigma
        #       ac1_optim = ac1

        # Initialise ac1 to optimal value
        #c1 = ac1_optim

        # Iterative procedure
        #or i in range(niter):

        #   # Compute OLS estimate with transformed variables
        #   P = self._gls_transform_matrix(nsamp, ac1)
        #   Xs = np.dot(P, X)
        #   Ys = np.dot(P, Y)
        #   tXXinvs = np.linalg.inv(np.dot(Xs.T,Xs))
        #   params, sigma, df = self._ols(tXXinvs, Xs, Ys)
        #   pp = params['estimate'].reshape((params.shape[0], 1))

        #   # Correct bias
        #   pp[0,0] = np.mean(Y-np.dot(X[:,1:], pp[1:]))

        #   # Store data
        #   params_gls_iter[:-1,i] = pp[:,0]

        #   # Estimate auto-correlation of residuals
        #   residuals = Y-np.dot(X, pp)
        #   tXXinvs = 1./(np.dot(residuals[:-1].T,residuals[:-1]))
        #   ac1params, ac1sigma, ac1df = self._ols(tXXinvs, residuals[:-1], residuals[1:])
        #   ac1 = ac1params['estimate'].values[0]
        #   params_gls_iter[-1,i] = ac1

        #   # Check convergence
        #   if i>0:
        #       delta = np.abs(ac1-params_gls_iter[-1,i-1])

        #       if delta < self.gls_epsilon:
        #           break

        return params, phi, sigma

    def getresiduals(self, Y, Yhat):

        # Compute residuals
        residuals = Y-Yhat

        # Extract innovation from AR1 if GLS AR1 
        if self.type == 'gls_ar1':
            r = nresiduals.reshape((len(residuals),))
            residuals = sutils.ar1inverse([self.phi, 0.], r)

        return residuals

    def diagnostic(self, Y, Yhat):
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
        b = np.mean(Y-Yhat)/mY

        # Coeff of determination
        d = 1-np.sum((Y-Yhat)**2)/np.sum((Y-mY)**2)

        # Ratio of variances
        rv = np.var(Yhat)/np.var(Y)

        # Store data
        diag = {'bias':b, 
            'coef_determination':d,
            'ratio_variance':rv,
            'shapiro_stat': s[0], 
            'shapiro_pvalue':s[1],
            'durbinwatson_stat': dw, 
            'R2': R2}

        return diag


    def fit(self):
        ''' Run parameter estimation and compute diagnostics '''

        # Fit
        if self.type == 'ols':
            params, sigma, df = self._ols()
            phi = None

        elif self.type =='gls_ar1':
            params, phi, sigma = self._gls_ar1()
            df = None

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

        return params, phi, sigma

    def getresiduals(self, Y, Yhat):

        # Compute residuals
        residuals = Y-Yhat

        # Extract innovation from AR1 if GLS AR1 
        if self.type == 'gls_ar1':
            r = residuals.reshape((len(residuals),))
            residuals = sutils.ar1inverse([self.phi, 0.], r)

        return residuals

    def diagnostic(self, Y, Yhat):
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
        b = np.mean(Y-Yhat)/mY

        # Coeff of determination
        d = 1-np.sum((Y-Yhat)**2)/np.sum((Y-mY)**2)

        # Ratio of variances
        rv = np.var(Yhat)/np.var(Y)

        # Store data
        diag = {'bias':b, 
            'coef_determination':d,
            'ratio_variance':rv,
            'shapiro_stat': s[0], 
            'shapiro_pvalue':s[1],
            'durbinwatson_stat': dw, 
            'R2': R2}

        return diag


    def fit(self):
        ''' Run parameter estimation and compute diagnostics '''

        # Fit
        if self.type == 'ols':
            params, sigma, df = self._ols()
            phi = None

        elif self.type =='gls_ar1':
            params, phi, sigma = self._gls_ar1()
            df = None

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
        diag = self.diagnostic(self.Y, self.Yhat)
        self.diag = diag

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

    #def boot(self, nsample=5000):
    #    ''' Confidence interval based on bootstrap '''

    #    nx = self.X.shape[0]
    #    for i in range(nsample):
    #        kk = np.random.randint(nx, size=nx)
    #        XB = self.X[kk,:]    
    #        YB = self.Y[kk]
    #        tXXinvB = np.linalg.inv(np.dot(XB.T,XB))
    #        params, Yhat, residuals, sigma = _olspars(tXXinvB, XB, YB)

    #    return 0
            
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
        ax.annotate(r'$R^2$ = %0.2f' % self.diag['R2'], 
                xy=(0.05, 0.93), xycoords='axes fraction')
