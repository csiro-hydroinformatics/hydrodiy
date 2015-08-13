from __future__ import generators

import os
from math import sqrt, exp, log 

import numpy as np

from scipy.linalg import cholesky as chol
from scipy.linalg import LinAlgError
from scipy.optimize import fmin
from scipy.optimize import approx_fprime

import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm 

import pandas as pd

from hyplot.putils import get_colors
from hyio.csv import write_csv, read_csv

def wrap_arg_function(function, fargs):
    ''' Function wrapper from scipy package '''
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] +=1
        return function(*(wrapper_args + fargs))
    
    return ncalls, function_wrapper

def neg_function(function, fargs):
    ''' Return -1 times function output '''
    if function is None:
        return None

    def function_wrapper(*wrapper_args):
        return -1. * function(*(wrapper_args + fargs))
    
    return function_wrapper


def mvt(means, Lcov):
    ''' 
        Get one sample from multivariate normal. 
        (this is not made to fast, just accurate) 
    
        :param numpy.array means: vector of means
        :param numpay.array Lcov: Cholesky decomposition of covariance matrix
                can be optained by applying 
                scipy.linalg.cholesky on covariance matrix
                Important: use option lower=true otherwise covariance
                won't be reproduced
        :returns: Array containing a sample from the multivariate normal
    '''
    nv = len(means)
    u = np.random.normal(size=nv)
    if nv>1:
        s = means.flatten() + Lcov.dot(u)
    else:
        # degenerate to single variabel if nv=1
        s = means+u*Lcov

    return s


def metro_runner(logPost,  
                prop_mean, prop_cov, 
                ndisplay=0, 
                nchains=5, 
                nsample=1000,
                par_names_short = None,
                fargs=()):
    ''' 
        Sample from posterior distribution with 
        Metropolis Hastings algorithm
       
        :param object logPost: function returning log posterior 
        :param np.array prop_mean: Mean vector of proposal 
                                    distribution
        :param np.array prop_cov: Covariance matrix of proposal 
                                    distribution
        :param boolean ndisplay: Display progresses or not
        :param int nchains: Number of parallel chains
        :param int nsample: Number of sample 
        :param list par_names_short: Short names for parameters
        :param tuple fargs: Arguments to the log posterior function 
    '''

    # Initialise variables
    npars = len(prop_mean)
    dtype = dtype_chain(npars, par_names_short)
    count = 0
    Lcov = prop_cov
    if npars>1:
        Lcov = chol(prop_cov, lower=True)

    pars_current = [prop_mean + 0.0]*nchains

    # Wrap log posterior
    ncall, logPost = wrap_arg_function(logPost, fargs)

    # Compute starting value for log posterior
    lp_current = [logPost(pars) for pars in pars_current]

    # Initialise chains
    output = np.zeros(nchains, dtype=dtype)
    par_names = [cn for cn in output.dtype.names 
                            if cn.startswith('param')]
    output['chain'] = [i+1 for i in range(nchains)]

    # Run chains
    while count < nsample:
        count += 1

        if ndisplay>0 and count%ndisplay==0:
            print('\tmcmc run %5d/%5d'%(count, nsample))

        for c in range(nchains):
            output['count'][c] = count
            
            # New parameter set and log posterior
            pars_new = mvt(prop_mean, Lcov)
            lp_new = logPost(pars_new)
            u = log(np.random.rand())
            lp_diff = lp_new - output['logpost'][c]

            output['logpost_diff'][c] = lp_diff
            output['random'][c] = u
            output['accepted'] = 0

            # Rejection rule
            if lp_diff>u:
                pars_current[c] = pars_new + 0.0
                lp_current[c] = lp_new
                output['accepted'][c] = 1    

            # Store data
            output['logpost'][c] = lp_current[c]
            for k in range(npars):
                output[par_names[k]][c] = pars_current[c][k]

        yield output

def metro_mcmc(logPost, start, ndisplay=0, 
                nchains=5, 
                nsample=500,
                par_names_short = None,
                fargs=(), par_scale=1.1):

    '''
        Full MCMC sampler with maximisation of log posterior 
        and sampling with Metropolis Hasting algorithm

        :param object logPost: function returning log posterior 
        :param numpy.array start: Start parameter for log posterior optimisation
        :param boolean ndisplay: Display progresses or not
        :param int nchains: Number of parallel chains
        :param int nsample: Number of sample drawn 
        :param list par_names_short: Short names for parameters
        :param tuple fargs: Arguments to the log posterior function 
        :param float par_scale: parameter scaling for quadratic estimation of covariance
    '''

    # Get maximum of log posterior
    if ndisplay:
        print('    MCMC STEP 1/2 - Maximisation of log posterior')
    
    logPostNeg = neg_function(logPost, fargs)
    opt = fmin(logPostNeg, start, full_output=True, disp=ndisplay>0) 
    prop_mean = opt[0]
    npars = len(prop_mean)
    dtype = dtype_chain(npars, par_names_short)

    # Estimate variance
    prop_cov = quadratic_approx_covar(logPost, opt[0], 
                        -1.*opt[1], fargs, par_scale)
    
    if ndisplay:
        txt = [' %0.3f '%p for p in prop_mean]
        print('      logpost max : %s'%(''.join(txt)))
        txt = ['(%0.3f)'%p for p in np.diagonal(prop_cov)]
        print('                    %s'%(''.join(txt)))
    
    # run final sampling and discard firts half 
    if ndisplay>0:
        print('    MCMC STEP 2/2 - Final sampling (%d samples)'%(nsample*2))

    sampler = metro_runner(logPost, 
            prop_mean, prop_cov, ndisplay, nchains, 
            2*nsample, par_names_short, fargs)
    
    sample = np.empty(0, dtype=dtype)
    for smp in sampler:
        sample = np.hstack([sample, smp])
    sample = sample[nchains*nsample:]

    return sample, prop_mean, prop_cov

def dtype_chain(npars, par_names_short=None):
    ''' Returns the dtype variable corresponding to an MCMC chain state '''

    dtype = [('chain', int), ('count', int), 
            ('accepted', int), ('logpost',float), 
            ('logpost_diff', float), ('random', float)]
    if par_names_short is None:
        dtype += [('param%2.2d'%(k+1), float) for k in range(npars)]
    else:
        dtype += [('param_%s'%par_names_short[k], float) 
                            for k in range(len(par_names_short))]

    return dtype


def get_parnames(chain_data):
    ''' Get list of parameters from chain data '''
    return [cn for cn in chain_data.dtype.names 
                            if cn.startswith('param')]

def write_chain_data(chain_data, filename, comment):
    ''' Writes chain data to zipped csv file '''

    source_file = os.path.abspath(__file__)
    write_csv(pd.DataFrame(chain_data), filename, comment, source_file)

def read_chain_data(filename):
    ''' Reads zipped csv file containing chain data '''
    data, comment = read_csv(filename)
    par_names_short = [cn for cn in data.columns
                    if cn.startswith('param')] 
    dtype = dtype_chain(len(par_names_short), par_names_short)

    return data.to_records()

def quadratic_approx_covar(logPost, par0, lp0, fargs, par_scale):
    ''' Compute quadratic approximation of log posterior '''

    ncall, logPost = wrap_arg_function(logPost, fargs)
    # log post values
    par1 = par0*par_scale
    lp1 = logPost(par1)
    # Quadratic approx
    var = np.abs((par1-par0)**2/(lp1-lp0))

    return np.diag(var)

def get_convergence_metric(chain_data):
    ''' Compute the R score from Gelman and '''
  
    # MCMC data characteristics
    par_names = get_parnames(chain_data)
    n = chain_data.shape[0] 
    ichains = np.unique(chain_data['chain'])
    m = len(ichains)
    assert m>1

    W = []
    B = []

    # Loop through parameters
    for par in par_names:
        
        mean_all = 0.
        means = []
        vars = []

        for ic in ichains:
            # index of data in chain
            idx = chain_data['chain']==ic

            # mean and variance of parameter in chain
            mc = np.mean(chain_data[par][idx])
            vc = np.sum((chain_data[par][idx]-m)**2)/(n-1)

            # Store data
            means.append(mc)
            mean_all += mc
            vars.append(vc)

        # Mean of all chains
        mean_all /= m 

        # W and B factors
        W.append(np.mean(vars))
        B.append((n+0.)/(m-1) * np.sum((means-mean_all)**2))

    # R factor for all parameters
    return [sqrt((n-1.)/n+b/w/n) for w, b in zip(W, B)]

def get_acceptance_rate(chain_data):
    ''' Compute the acceptance rate '''
  
    return np.mean(chain_data['accepted'])


def get_means_and_covar(chain_data):
    ''' Compute means and covariance matrix of samples '''
    # compute means
    par_names = get_parnames(chain_data)
    par_data = chain_data[par_names].view(dtype=float)
    par_data = par_data.reshape((len(chain_data), len(par_names)))
    prop_mean = np.mean(par_data, axis=0)

    # Compute covariance. If fails revert to initial estimate
    try:
        prop_cov = np.cov(par_data.T) 
        Lcov = chol(prop_cov)
    except LinAlgError:
        prop_cov = None

    return prop_mean, prop_cov

def plot_chains(chain_data, fig, nval_trace=100):
    ''' 
        plot results of MCMC runs. Places axes on a matplotlib fig object
        :param np.array chain_data : Samples produced by metro_mcmc
        :param matplotlib.figure fig : Figure to draw on
        :param int nval_trace : Number of parameter to use in trace plot

        :returns list axs: Axes used to plot data
        :returns matplotlib.gridspec gs : Gridspec used to place axes 
                    e.g. ax = fig.add_subplot(gs[0, 0]) for upper left corner
    '''

    # MCMC chain data charateristics
    par_names = get_parnames(chain_data)
    npars = len(par_names)

    chain_id = np.unique(chain_data['chain'])
    nchains = len(chain_id)

    # metrics
    r_metric = get_convergence_metric(chain_data)
    acc = get_acceptance_rate(chain_data)
    means, cov = get_means_and_covar(chain_data)
    
    # Initialise plotting objects
    cols = get_colors(nchains+2, 'YlOrRd')[2:] 
    gs = gridspec.GridSpec(npars, 2*npars)
    fig.suptitle('Acceptance rate = %0.2f'%acc, fontsize=18)
    ylim = []
    axs = dict()

    for i in range(npars):
        ylim.append((np.min(chain_data[par_names[i]]),
                np.max(chain_data[par_names[i]])))

        # Traces plots
        ax = fig.add_subplot(gs[i,:npars])
        all_par_values = []
        for chain in chain_id:
            idx = chain_data['chain']==chain
            par_values = chain_data[par_names[i]][idx]
            all_par_values.append(par_values)
            ax.plot(par_values[-nval_trace:], color=cols[chain-1])

        ax.set_ylim(ylim[i])
        ax.grid()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('%s values'%par_names[i])
        ax.set_title('%s / last %d values / R=%0.3f'%(par_names[i], 
                        nval_trace, r_metric[i]))
        
        axs['trace_%s'%par_names[i]] = ax

        # Histograms plots
        ax = fig.add_subplot(gs[i,i+npars])
        ax.hist(all_par_values, color=cols,
                alpha=0.5, orientation='horizontal')
        ax.grid()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('%s values'%par_names[i])
        ax.set_title('mean=%0.2f std=%0.2f'%(means[i], sqrt(cov[i,i])))
        ax.set_ylim(ylim[i])

        axs['hist_%s'%par_names[i]] = ax

    # Correlation plots
    if npars>1:
        for i in range(1, npars):
            y = chain_data[par_names[i]]
            nval = len(y)
            
            for j in range(i-1, npars-1):
                x = chain_data[par_names[j]]
                
                ax = fig.add_subplot(gs[i,j+npars])
                ax.hist2d(x, y, bins=40, norm=LogNorm(), cmap='YlOrBr')
                ax.grid()
                ax.set_ylim(ylim[i])
                ax.set_ylabel('%s values'%par_names[i])
                ax.set_xlabel('%s values'%par_names[j])

                axs['correlation_%s_%s'%(par_names[i], par_names[j])] = ax

    return axs, gs
