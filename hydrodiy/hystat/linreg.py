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

    def __init__(self, x, y, intercept=True, polyorder=1, type='ols'):
        ''' Initialise regression model

            :param np.array x: Predictor variable
            :param np.array  y: Predictand variable
            :param bool intercep: Add intercept to regression model?
            :param int polyorder: Add polynomial terms with higher order (polyorder>1)

        '''
        # build input and output matrix
        self.intercept = intercept
        self.polyorder = polyorder
        
        npts = len(y)
        xx = np.array(x)
        self.nvar = np.prod(xx.shape)/npts

        X = self._buildinput(xx, npts)
        self.tXXinv = np.linalg.inv(np.dot(X.T,X))
        Y = np.array(y).reshape((npts, 1))

        # run parameter estimation
        self.type = type
        if self.type == 'ols':
            params, sigma, df = self._ols(self.tXXinv, X, Y)
            pgi = None
        if self.type =='gls_ar1':
            params, sigma, df, pgi = self._gls_ar1(X, Y)

        self.params = params
        self.params_gls_iter = pgi
        self.sigma = sigma
        self.df = df

        # compute fit
        fit = np.dot(X, self.params['estimate'])
        fit = fit.reshape((npts, 1))
        self._check(Y, fit)
    
    def __str__(self):
        str = '\t** Linear model **\n'
        str += '\n\tModel setup:\n'
        str += '\t  Type: %s\n'%self.type
        str += '\t  Intercept: %s\n'%self.intercept
        str += '\t  Polynomial order: %d\n\n'%self.polyorder
        str += '\tParameters:\n'
        for idx, row in self.params.iterrows():
            str += '\t  params %d = %5.2f [%5.2f,%5.2f] P(>|t|)=%0.3f\n'%(idx, 
                row['estimate'], row['confint_025'], 
                row['confint_975'], row['Pr(>|t|)'])

        str += '\n\tPerformance:\n'
        str += '\t  R2 = %0.3f\n'%self.R2

        str += '\n\tTest on normality of residuals (Shapiro):\n'
        sh = self.shapiro_residuals['p-value']
        mess = ''
        if sh<0.05: mess = '(<0.05 : failing normality at 5% level)'
        str += '\t  P value = %0.3f %s\n'%(sh, mess)

        str += '\n\tTest on independence of residuals (Durbin-Watson):\n'
        dw = self.durbin_watson_residuals['stat']
        mess = ''
        if dw<1: mess = '(<1 : residuals may not be independent)'
        str += '\t  Statistic = %0.3f %s\n'%(dw, mess)
        
        return str

    def _buildinput(self, xx, npts):
        ''' Build regression input matrix '''
        XX = xx.reshape(npts, self.nvar)
        assert self.polyorder>0

        X = np.empty((XX.shape[0], XX.shape[1] * self.polyorder), float)
        for j in range(XX.shape[1]):
            for k in range(self.polyorder):
                X[:, j*self.polyorder+k] = XX[:, j]**(k+1)

        if self.intercept:
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

    def _gls_ar1(self, X, Y, niter=5):
        ''' Estimate parameter with generalized least squares 
            assuming ar1 residuals
        '''
        npts = X.shape[0]
        P = np.eye(npts)
        params_gls_iter = np.empty((X.shape[1], niter))

        for i in range(niter):
            # OLS estimate with transformed variables
            Xs = np.dot(P, X)
            Ys = np.dot(P, Y)
            tXXinvs = np.linalg.inv(np.dot(Xs.T,Xs))
            params, sigma, df = self._ols(tXXinvs, Xs, Ys)
            pp = params['estimate'].reshape((params.shape[0], 1))
            params_gls_iter[:,i] = pp[:,0]

            if i<niter-1:
                # Estimate auto-correlation of OLS residuals
                residuals = Y-np.dot(X, pp)
                ac1 = sutils.acf(residuals, lag=[1])['acf'].values[0]

                # Compute transformation matrix
                P = np.eye(npts)
                P[0,0] = math.sqrt(1-ac1**2)
                P -= np.diag([1]*(npts-1), -1)*ac1

        return params, sigma, df, params_gls_iter

    def predict(self, x0, coverage=[95, 80]):
        ''' Pediction with intervals 
            
            :param numpy.array x0: regression input
            :param list quantiles: Coverage of prediction intervals
        '''

        if self.type!='ols':
            raise ValueError('Only predict output for ols regressions')

        xx0 = np.array(x0)
        npts = xx0.shape[0]
        X0 = self._buildinput(xx0, npts) 
        Y0 = np.dot(X0, self.params['estimate'])
        nq = len(coverage)*2
        
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

    def _check(self, Y, fit):
        ''' perform tests on assumptions '''
        residuals = Y-fit

        # Shapiro Wilks on residuals
        s = shapiro(residuals)
        self.shapiro_residuals = {'stat':s[0], 'p-value':s[1]}

        # Durbin watson test
        res = residuals[:,0]
        de = np.diff(res, 1)
        dw = np.dot(de, de)/np.dot(res, res)
        self.durbin_watson_residuals = {'stat':dw, 'p-value':np.nan}

        # correlation
        u = Y-np.mean(Y)
        v = fit-np.mean(fit)
        self.R2 = np.sum(u*v)**2/np.sum(u**2)/np.sum(v**2)

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
    #        
    #def plot(self):
    #    ''' plot some common data '''


