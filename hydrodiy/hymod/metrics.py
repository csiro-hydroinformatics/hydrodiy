       
import re

import numpy as np
import pandas as pd

from scipy.special import kolmogorov
from scipy.stats import kendalltau

import c_hymod

from hystat import sutils

try:
    from fcvf import skillscore,pits
    HAS_FCVF = True
except ImportError:
    HAS_FCVF = False


def __checkdims(obs, ens, ref):

    nforc = ens.shape[0]

    if len(obs)!=nforc:
        raise ValueError('Length of obs(%d) different from length of ens(%d)'%(len(obs), nforc))

    if ref.shape[0]!=nforc:
        raise ValueError('Length of ref(%d) different from length of ens(%d)'%(ref.shape[0], nforc))

    return nforc


def hypit(yobs,ysim,has_ties=True,pp_cst=0.3):
    """

    Compute PIT values in the case of ties
        
    yobs    obs values, [n] or [n,1] np.array
    ysim    sim values, [n] or [n,1] np.array
    has_ties Are there ties in ysim ? Default is True.
    pp_cst  constant used to compute plotting positions

    """
    ys_sort = np.sort(ysim)

    # if there are ties, adds small number to ysim
    # and sort again
    if has_ties:
        max_dy = np.max(np.diff(ys_sort))
        ys_sort = np.sort(ysim + max_dy * 1e-10 * 
                        np.random.random(len(ys_sort)))

    prob_obs = 0.0
    if yobs>= ys_sort[-1]: prob_obs = 1.0
    elif yobs>= ys_sort[0]:
        delta = 1/(len(ys_sort)+1-2*pp_cst)
        unif_dist = (np.arange(1, len(ys_sort)+1)-pp_cst)*delta
        prob_obs = np.interp(yobs,ys_sort,unif_dist)

    assert 0.0 <= prob_obs <= 1.0

    return prob_obs

def cut(ysim, cats):
    ''' Convert probabilistic forecasts into categorical forecasts
        using a common set of categories
 
    Parameters
    -----------
    ysim : numpy.ndarray
        Probabilistic forecasts stored in an NxP matrix.
        N = Number of forecasts
        P = Number of ensembles for each forecasts
    cats : list
        Values defining the bounds of forecast categories (M values)

    Returns
    -----------
    ysim_cat : numpy.ndarray
        
    Example
    -----------
    >>> import numpy as np
    >>> np.random.seed(333)  
    >>> N = 100; P = 500
    >>> ysim = np.random.uniform(0, 10, size=(N, P))
    >>> cats = [0, 2, 7, 7]
    >>> metrics.cut(ysim, cats) 

    '''

    # Add -infty and +infty boundaries
    cats = np.array([[-np.infty]+list(np.sort(cats))+[np.infty]])
    cats = cats.flatten()

    # Ensemble data
    if len(ysim.shape)>1:
        # Special cut functions for ensemble data
        def cutfun(x):
            xc = pd.cut(x, cats)
            return x.groupby(xc).apply(lambda u:float(len(u))/x.shape[0])

        # Compute categorical data
        ysim_cat = pd.DataFrame(ysim).apply(cutfun, axis=1)

    # Deterministic data
    else:
        # Compute categorical data
        ysimc = pd.cut(ysim, cats)

        ysimcd = pd.DataFrame({
            'idx':range(ysim.shape[0]), 
            'lab':ysimc, 
            'count':1
        })

        ysim_cat =  pd.pivot_table(ysimcd, index='idx', columns='lab', values='count')

        # Reorder columns
        v2 = [float(re.sub('.*,|\\]', '', cn)) for cn in ysim_cat.columns]
        kk = np.argsort(v2)
        ysim_cat = ysim_cat.iloc[:, kk]

        # Replace NaN with 0
        ysim_cat = ysim_cat.fillna(0)


    return ysim_cat


def drps(yobs, ysim, cats):
    ''' Compute the DRPS score with categories defined by cats '''

    nforc = __checkdims(yobs, ysim, yobs)

    # find unique cats
    cats_order = np.lexsort(cats.T)
    cats_sort = cats[cats_order]

    diff = np.diff(cats_sort, axis=0)
    ui = np.ones(len(cats), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    cats_unique = cats_sort[ui]
    

    # loop through unique cats
    drps_all = []
    drps_value = 0.
    for ca in cats_unique:
        idx = np.sum(cats==ca, axis=1) == cats.shape[1]
        yobsc_idx = cut(yobs[idx], ca)
        ysimc_idx = cut(ysim[idx], ca)

        d = (yobsc_idx-ysimc_idx)**2

        drps_all.append({
            'drps_value':d.sum(axis=1).mean(), 
            'idx':idx, 
            'cats':ca, 
            'nval':np.sum(idx)
        })

        drps_value += d.sum(axis=1).sum()

    drps_value /= nforc

    return drps_value, drps_all


def crps(yobs, ysim):
    ''' Compute the CRPS decomposition from Hersbach (2000) '''

    __checkdims(yobs, ysim, yobs)

    nval = ysim.shape[0]
    ncol = ysim.shape[1]

    # set weights to zero and switch off use of weights
    weights = np.zeros(nval)
    use_weights = 0
    
    # run C code via cython
    reliab_table = np.zeros((ncol+1, 7))
    crps_decompos = np.zeros(5)
    is_sorted = 0
    ierr = c_hymod.crps(use_weights, is_sorted, yobs, ysim, 
                    weights, reliab_table, crps_decompos)

    reliab_table = pd.DataFrame(reliab_table)
    reliab_table.columns = ['freq', 'a', 'b', 'g', 'rank', 
                'reliability', 'crps_potential']

    d = crps_decompos
    crps_decompos= {
        'crps':d[0],
        'reliability':d[1],
        'resolution':d[2],
        'uncertainty':d[3],
        'crps_potential':d[4]
    }

    return crps_decompos, reliab_table

def alpha(yobs, ysim, pp_cst = 0.3):
    ''' Score computing the Pvalue of the Kolmogorov-Smirnov test '''

    __checkdims(yobs, ysim, yobs)

    pit = [hypit(o,s,pp_cst) 
                    for o,s in zip(yobs,ysim)] 
    nval = len(pit)
    uniform_dens = (np.arange(nval)+1-pp_cst)/(nval+1-2*pp_cst)
    distances = np.sort(pit)-uniform_dens
    max_dist = np.max(np.abs(distances))

    return kolmogorov(np.sqrt(nval)*max_dist)


def iqr_scores(obs, ens, ref, coverage=50.):
    ''' Compute the interquantile range (iqr) divided by clim and iqr reliability'''

    nforc = __checkdims(obs, ens, ref)

    nens = ens.shape[1]

    iqr = np.zeros((nforc,3))
    iqr_clim = np.zeros((nforc,3))
    rel = np.zeros(nforc)
    rel_clim = np.zeros(nforc)
    
    perc =[coverage/2, 100.-coverage/2]

    for i in range(nforc):
        iqr_clim[i, :2] = sutils.percentiles(ref[i,:], perc)
        iqr_clim[i, 2] = iqr_clim[i,1]-iqr_clim[i,0]

        iqr[i, :2] = sutils.percentiles(ens[i,:], perc)
        iqr[i, 2] = iqr[i,1]-iqr[i,0]

        rel[i] = int( (obs[i]>=iqr[i,0]) & (obs[i]<=iqr[i,1]) )
        rel_clim[i] = int( (obs[i]>=iqr_clim[i,0]) & (obs[i]<=iqr_clim[i,1]) )

    pre_sc = np.mean(iqr[:,2]/iqr_clim[:, 2])
    rel_sc = np.mean(rel)
    rel_clim_sc = np.mean(rel_clim)
        
    out = {'precision_skill': (1-pre_sc)*100,
             'reliability_skill': (1-abs(rel_clim_sc-rel_sc)/rel_clim_sc)*100, 
             'precision_score': pre_sc,
             'reliability_score': rel_sc}

    return out, iqr, iqr_clim, rel, rel_clim


def median_contingency(obs, ens, ref):
    ''' Compute the contingency matrix for below/above median forecast.
        A positive event is equivalent to below median (i.e. dry)

        Contigency matrix is presented as follows:
        ----------------------------------------------
        | obs<med & ens<med   | obs<med & ens>=med   |
        |--------------------------------------------|
        | obs>=med & ens<med  | obs>=med & ens>=med  |
        ----------------------------------------------
    '''

    nforc = __checkdims(obs, ens, ref)

    cont = np.zeros((2,2))
    medians = np.zeros(nforc)

    for i in range(nforc):
        obs_med = sutils.percentiles(ref[i,:], 50.).values[0]

        medians[i] = obs_med
        med_obs = int(obs[i]>= obs_med)

        umed = np.mean(ens[i,:]>= obs_med)
        cont[med_obs, int(round(umed))] += 1

    hit = (cont[0,0] + cont[1,1] +0.)/np.sum(cont)
    miss_low = (0.+cont[1,0])/np.sum(cont[:,0])

    return cont, hit, miss_low, medians


def tercile_contingency(obs, ens, ref):
    ''' Compute the contingency matrix for below/above terciles forecast

        Contigency matrix is presented as follows:
        ----------------------------------------------------------------------------------------
        | obs<t1 & ens<t1         | obs<t1 & ens in [t1,t2[         | obs<t1 & ens>=t2         |
        ----------------------------------------------------------------------------------------
        | obs in [t1,t2[ & ens<t1 | obs in [t1,t2[ & ens in [t1,t2[ | obs in [t1,t2[ & ens>=t2 |
        ----------------------------------------------------------------------------------------
        | obs >= t2 & ens<t1      | obs >= t2 & ens in [t1,t2[      | obs>=t2 & ens>=t2        |
        ----------------------------------------------------------------------------------------

        hit_low is the ratio between the top left cell and the total of the first column
        (provided low value is forecasted, how many times low value occurs)

        hit_high is the ratio between the bottom right cell and the total of the last column
        (provided high value is forecasted, how many times high value occurs)

    ''' 

    nforc = __checkdims(obs, ens, ref)

    cont = np.zeros((3,3))
    terciles = np.zeros((nforc, 2))

    for i in range(nforc):
        obs_t1 = sutils.percentiles(ref[i,:], 100./3).values[0]
        obs_t2 = sutils.percentiles(ref[i,:], 200./3).values[0]
        terciles[i,:] = [obs_t1, obs_t2]
    
        t_obs = (obs[i]>= obs_t1).astype(int) + (obs[i]>=obs_t2).astype(int)
        uu = (ens[i,:] >= obs_t1).astype(int) + (ens[i,:] >= obs_t2).astype(int)
        ut = np.bincount(uu)
        cont[t_obs, np.argmax(ut)] += 1

    hit = (cont[0,0] + cont[1,1] + cont[2,2] + 0.)/np.sum(cont)
    hit_low = (cont[0,0] + 0.)/np.sum(cont[:,0]) 
    hit_high = (cont[2,2] + 0.)/np.sum(cont[:,2])
    miss_low = (0.+np.sum(cont[1:,0]))/np.sum(cont[:,0])

    return cont, hit, miss_low, hit_low, hit_high, terciles


def det_metrics(yobs,ysim, compute_persistence=False, min_val=0., eps=1):
    """
        Compute deterministic performance metrics
        
        :param np.array yobs: Observations
        :param np.array ysim: Simulated data
        :param float yref: Reference flow value
        :param bool compute_persistence: Compute persistence metrics
                (will restrict data valid data)
        :param float min_val: Threshold below which data is considered missing
        :param float eps: Value added to 0 when computing log

    """

    nforc = __checkdims(yobs, ysim, yobs)

    # inputs
    yobs = np.array(yobs, copy=False)
    yobs_shift = np.roll(yobs, 1)
    yobs_shift[0] = np.nan
    ysim = np.array(ysim, copy=False).flatten()

    idx = np.isfinite(yobs) & np.isfinite(ysim) 
    idx = idx & (yobs>=min_val) & (ysim>=min_val)
    if compute_persistence:
        idx = idx & (yobs_shift>=min_val) & np.isfinite(yobs_shift)

    e = ysim[idx] - yobs[idx]
    elog = np.log(eps+ysim[idx]) - np.log(eps+yobs[idx])
    einv = 1/(eps+ysim[idx]) - 1/(eps+yobs[idx])
    if compute_persistence:
        esh = yobs_shift[idx] - yobs[idx]
        esh_inv = 1/(eps+yobs_shift[idx]) - 1/(eps+yobs[idx])
    
    # Obseved mean and variance
    mo = np.mean(yobs[idx])
    vo = np.var(yobs[idx])

    molog = np.mean(np.log(eps+yobs[idx]))
    volog = np.var(np.log(eps+yobs[idx]))

    moinv = np.mean(1/(eps+yobs[idx]))
    voinv = np.var(1/(eps+yobs[idx]))
    
    # metrics
    nse = 1.0 - np.mean(np.square(e))/vo
    nselog = 1.0 - np.mean(np.square(elog))/volog
    nseinv = 1.0 - np.mean(np.square(einv))/voinv
    corr = np.corrcoef(yobs[idx],ysim[idx],rowvar=1)[0,1]
    bias = np.mean(e)/mo
    biaslog = np.mean(elog)/molog
    biasinv = np.mean(einv)/moinv
    ratiovar = np.var(ysim[idx])/vo

    tau, p_value = kendalltau(yobs, ysim)
   
    persist = np.nan
    persist_inv = np.nan
    if compute_persistence:
        persist = 1.0 - np.mean(np.square(e))/np.mean(np.square(esh))
        persist_inv = 1.0 - np.mean(np.square(einv))/np.mean(np.square(esh_inv))

    metrics  = {
            'nse':nse, 
            'nselog':nselog, 
            'nseinv':nseinv,
            'persist':persist, 
            'persist_inv':persist_inv, 
            'nseinv':nseinv, 
            'bias':bias, 
            'biaslog':biaslog, 
            'biasinv':biasinv,
            'bias_skill':(1-abs(bias))*100, 
            'biaslog_skill':(1-abs(biaslog))*100, 
            'biasinv_skill':(1-abs(biasinv))*100,
            'corr':corr, 'ratiovar':ratiovar, 
            'kendalltau_skill':tau*100,
            'kendalltau':tau
    }

    return metrics, idx


def ens_metrics(yobs,ysim, yref=None, pp_cst=0.3, min_val=0.):
    """
        Computes a set of ensemble performance metrics
        
        :param np.array yobs: Observations (n values)
        :param np.array ysim: Simulated ensemble data (n x p values)
        :param np.array ysim: Reference data (1 ensemble with m values)
        :param float pp_cst:  Constant used to compute plotting position
        :param float min_val: Threshold below which data is considered missing

    """

    # inputs
    yobs = np.array(yobs, copy=False)
    ysim = np.array(ysim, copy=False)
    nval = len(yobs)

    if yref is None: 
        yref = np.array([yobs]*nval)

    __checkdims(yobs, ysim, yref)

    # find proper data
    idx = np.isfinite(yobs)
      
    # alpha score
    al = alpha(yobs[idx], ysim[idx,:])

    # crps
    cr, rt = crps(yobs[idx],ysim[idx,:])

    # iqr
    iqr = iqr_scores(yobs[idx], ysim[idx,:], yref[idx,:])

    # contingency tables
    cont_med, hit_med, miss_medlow, medians = median_contingency(yobs[idx], ysim[idx,:], yref[idx,:])
    cont_terc, hit_terc, miss_terclow, hit_terclow, hit_terchigh, terciles = tercile_contingency(yobs[idx], 
            ysim[idx,:], yref[idx,:])

    # drps
    dr, dr_all = drps(yobs[idx], ysim[idx,:], cats=terciles)
    dr_clim, dr_all = drps(yobs[idx], yref[idx,:], cats=terciles)
    dr_skill = (1-dr/dr_clim)*100

    # FCVF skill scores
    rmse_fcvf = np.repeat(np.nan, 3)
    rmsep_fcvf = rmse_fcvf
    crps_fcvf = rmse_fcvf

    if HAS_FCVF:
        crps_fcvf = skillscore.crps(yobs[idx],ysim[idx,:],yref[idx])
        rmse_fcvf = skillscore.rmse(yobs[idx],ysim[idx,:],yref[idx])
        rmsep_fcvf = skillscore.rmsep(yobs[idx],ysim[idx,:],yref[idx])

    # Compile all metrics
    metrics = {
            'nval':nval, 
            'nval_ok':np.sum(idx), 
            'alpha': al,
            'iqr_precision_skill': iqr[0]['precision_skill'],
            'iqr_reliability_skill': iqr[0]['reliability_skill'],
            'iqr_precision_score': iqr[0]['precision_score'],
            'iqr_reliability_score': iqr[0]['reliability_score'],
            'median_hitrates':hit_med,
            'median_missrateslow':miss_medlow,
            'tercile_hitrates':hit_terc,
            'tercile_hitrateslow':hit_terclow,
            'tercile_hitrateshigh':hit_terchigh,
            'tercile_missrateslow':miss_terclow,
            'drps': dr,
            'drps_skill': dr_skill,
            'crps': cr['crps'],
            'crps_potential': cr['crps_potential'],
            'crps_uncertainty': cr['uncertainty'],
            'crps_reliability': cr['reliability'],
            'crps_skill': (1-cr['crps']/cr['uncertainty'])*100,
            'crps_potential_skill' : (1-cr['crps_potential']/cr['uncertainty'])*100,
            'crps_reliability_skill' : (1-cr['reliability']/cr['uncertainty'])*100,
            'crps_skill_fcvf': crps_fcvf[0],
            'rmse_skill_fcvf': rmse_fcvf[0],
            'rmsep_skill_fcvf': rmsep_fcvf[0],
            'crps_score_fcvf': crps_fcvf[1],
            'rmse_score_fcvf': rmse_fcvf[1],
            'rmsep_score_fcvf': rmsep_fcvf[1],
            'crps_ref_fcvf': crps_fcvf[2],
            'rmse_ref_fcvf': rmse_fcvf[2],
            'rmsep_ref_fcvf': rmsep_fcvf[2]
    }

    return metrics, idx, rt, cont_med, cont_terc


def ens_metrics_names():
    
    o = np.random.lognormal(size=10)
    s = np.random.normal(size=(10,20))
    sc, idx, rt, cont_med, cont_terc = ensc_hymod(o, s)
    
    return sc.keys()


def det_metrics_names():
    
    o = np.random.lognormal(size=10)
    s = np.random.normal(size=(10,1))
    sc, idx = detc_hymod(o, s)
    
    return sc.keys()

