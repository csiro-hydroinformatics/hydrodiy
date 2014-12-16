import math

import numpy as np
import pandas as pd
from scipy.stats import t as student
from scipy.stats import shapiro
from scipy.linalg import circulant
from scipy.linalg import cholesky
from scipy.linalg import tril

from hystat import sutils

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

    def fit(self):
        ''' Run parameter estimation and compute diagnostics '''
 
        # build input and output matrix
        x = self.x
        y = self.y
        npts = len(y)
        xx = np.array(x)
        self.npredictors = np.prod(xx.shape)/npts

        X = self._buildinput(xx, npts)
        self.tXXinv = np.linalg.inv(np.dot(X.T,X))

        Y = np.array(y).reshape((npts, 1))
        self.npredictands = Y.shape[1]

        # Fit
        if self.type == 'ols':
            params, sigma, df = self._ols(self.tXXinv, X, Y)
            pgi = None
            ar1 = None

        elif self.type =='gls_ar1':
            params, ar1, sigma, df, pgi = self._gls_ar1(X, Y)

        else:
            raise ValueError('Regression type %s not recognised' % type)

        # Store data
        self.params = params
        self.params_gls_iter = pgi
        self.sigma = sigma
        self.df = df
        self.ar1 = ar1

        # compute fit
        fit = np.dot(X, self.params['estimate'])
        fit = fit.reshape((npts, 1))
        self._diagnostic(Y, fit)
    
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
            str += '\t  phi = %0.3f\n'%self.ar1
 
        str += '\n\tPerformance:\n'
        str += '\t  R2 = %0.3f\n'%self.R2

        str += '\n\tTest on normality of residuals (Shapiro):\n'
        sh = self.shapiro_residuals['p-value']
        mess = '(<0.05 : failing normality at 5% level)'
        str += '\t  P value = %0.3f %s\n'%(sh, mess)

        str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
        dw = self.durbin_watson_residuals['stat']
        mess = '(<1 : residuals may not be independent)'
        str += '\t  Statistic = %0.3f %s\n'%(dw, mess)
        
        return str

    def _buildinput(self, xx, npts):
        ''' Build regression input matrix, i.e. the matrix [1,xx] for standard regressions
            or [xx] if the intercept is not included in the regression equation.

        '''
        XX = xx.reshape(npts, self.npredictors)
        assert self.polyorder>0

        X = np.empty((XX.shape[0], XX.shape[1] * self.polyorder), float)
        for j in range(XX.shape[1]):
            for k in range(self.polyorder):
                X[:, j*self.polyorder+k] = XX[:, j]**(k+1)

        if self.has_intercept:
            X = np.insert(X, 0, 1., axis=1)

        return X

    def _ols(self, tXXinv, X, Y):
        ''' Estimate parameter with ordinary least squares '''

        # OLS parameter estimate
        pars = np.dot(tXXinv, np.dot(X.T, Y)) 
        Yhat = np.dot(X, pars)
        residuals = Y-Yhat
        df = Y.shape[0]-len(pars)
        sigma = math.sqrt(np.dot(residuals.T, residuals)/df)
        sigma_pars = sigma * np.sqrt(np.diag(tXXinv))

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

    def _gls_transform_matrix(self, npts, ac1):
        ''' Compute transformation matrix required for GLS iterative solution'''

        P = np.eye(npts)
        P[0,0] = math.sqrt(1-ac1**2)
        P -= np.diag([1]*(npts-1), -1)*ac1

        return P

    def _gls_ar1_loglikelihood(innov, sigma, phi):
        ''' Returns the log-likelihood of the GLS ar1 innovations '''
        n = float(len(innov))
        sse = np.sum(innov**2)
        ll = -n/2*math.log(math.pi)-n*math.log(sigma)+math.log(1-phi**2)/2-sse/(2*sigma)

        return ll

    def _gls_ar1(self, X, Y):
        ''' Estimate parameter with generalized least squares 
            assuming ar1 residuals
        '''

        niter = self.gls_niterations
        npts = X.shape[0]
        params_gls_iter = np.empty((X.shape[1]+1, niter))

        # Systematic exploration of initial AR1 values
        min_sigma = np.inf
        ac1_optim = 0.
        
        for ac1 in np.linspace(-0.99, 0.99, self.gls_nexplore):
            P = self._gls_transform_matrix(npts, ac1)
            Xs = np.dot(P, X)
            Ys = np.dot(P, Y)
            tXXinvs = np.linalg.inv(np.dot(Xs.T,Xs))
            params, sigma, df = self._ols(tXXinvs, Xs, Ys)

            if sigma < min_sigma:
                min_sigma = sigma
                ac1_optim = ac1

        # Initialise ac1 to optimal value
        ac1 = ac1_optim

        # Iterative procedure
        for i in range(niter):

            # Compute OLS estimate with transformed variables
            P = self._gls_transform_matrix(npts, ac1)
            Xs = np.dot(P, X)
            Ys = np.dot(P, Y)
            tXXinvs = np.linalg.inv(np.dot(Xs.T,Xs))
            params, sigma, df = self._ols(tXXinvs, Xs, Ys)
            pp = params['estimate'].reshape((params.shape[0], 1))

            # Correct bias
            pp[0,0] = np.mean(Y-np.dot(X[:,1:], pp[1:]))

            # Store data
            params_gls_iter[:-1,i] = pp[:,0]

            # Estimate auto-correlation of residuals
            residuals = Y-np.dot(X, pp)
            tXXinvs = 1./(np.dot(residuals[:-1].T,residuals[:-1]))
            ac1params, ac1sigma, ac1df = self._ols(tXXinvs, residuals[:-1], residuals[1:])
            ac1 = ac1params['estimate'].values[0]
            params_gls_iter[-1,i] = ac1

            # Check convergence
            if i>0:
                delta = np.abs(ac1-params_gls_iter[-1,i-1])

                if delta < self.gls_epsilon:
                    break

        return params, ac1, sigma, df, params_gls_iter

    def predict(self, x0=None, coverage=[95, 80]):
        ''' Pediction with intervals 
            
            :param numpy.array x0: regression input
            :param list quantiles: Coverage of prediction intervals
        '''

        # Prediction data
        if x0 is None:
            x0 = self.x
        xx0 = np.array(x0)
        
        if len(xx0.shape) == 1:
            npred = 1
        else:
            npred = xx0.shape[-1]

        if npred != self.npredictors:
            raise ValueError('Number of predictors in input data(%d) different from regression(%d)' %(npred,
                self.npredictors))

        npts = xx0.shape[0]
        X0 = self._buildinput(xx0, npts) 
        Y0 = np.dot(X0, self.params['estimate'])
        nq = len(coverage)*2
       
        # Prediction intervals (only for OLS)
        PI = None

        if self.type == 'ols':
            PI = pd.DataFrame(np.zeros((npts ,nq)))
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

    def _getresiduals(self, Y, fit):

        # Compute residuals
        residuals = Y-fit

        # Extract innovation from AR1 if GLS AR1 
        if self.type == 'gls_ar1':
            r = residuals.reshape((len(residuals),))
            residuals = sutils.ar1inverse([self.ar1, 0.], r)

        return residuals

    def _diagnostic(self, Y, fit):
        ''' perform tests on regression assumptions '''

        residuals = self._getresiduals(Y, fit)

        # Shapiro Wilks on residuals
        s = shapiro(residuals)
        self.shapiro_residuals = {'stat':s[0], 'p-value':s[1]}

        # Durbin watson test
        residuals = residuals.reshape((len(residuals), ))
        de = np.diff(residuals, 1)
        dw = np.dot(de, de)/np.dot(residuals, residuals)
        self.durbin_watson_residuals = {'stat':dw, 'p-value':np.nan}

        # correlation
        u = Y-np.mean(Y)
        v = fit-np.mean(fit)
        self.R2 = np.sum(u*v)**2/np.sum(u**2)/np.sum(v**2)

    def boot(self, nsample=5000):
        ''' Confidence interval based on bootstrap '''

        nx = self.X.shape[0]
        for i in range(nsample):
            kk = np.random.randint(nx, size=nx)
            XB = self.X[kk,:]    
            YB = self.Y[kk]
            tXXinvB = np.linalg.inv(np.dot(XB.T,XB))
            params, Yhat, residuals, sigma = _olspars(tXXinvB, XB, YB)

        return 0
    #        
    #def plot(self):
    #    ''' plot some common data '''


