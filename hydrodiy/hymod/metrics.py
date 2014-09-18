import numpy as np
from scipy.special import kolmogorov
import _metrics
from hystat import sutils

try:
    from fcvf import skillscore,pits
    HAS_FCVF = True
except ImportError:
    HAS_FCVF = False


def skill(score, ref, perfect):
    return (score-ref)/(perfect-ref)

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

def crps(yobs, ysim):
    ''' Compute the CRPS decomposition from Hersbach (2000) '''

    nval = ysim.shape[0]
    ncol = ysim.shape[1]

    # set weights to zero and switch off use of weights
    weights = np.zeros(nval)
    use_weights = 0
    
    # run C code via cython
    reliab_table = np.zeros((ncol+1, 7))
    crps_decompos = np.zeros(5)
    is_sorted = 0
    ierr = _metrics.crps(use_weights, is_sorted, yobs, ysim, 
                    weights, reliab_table, crps_decompos)

    reliab_table.dtype = np.dtype([('freq','double'),
                ('a','double'), ('b','double'),
                ('g','double'), ('ensemble_rank','double'),
                ('reliability','double'), ('crps_potential','double')])

    crps_decompos.dtype = np.dtype([('crps','double'),
                ('reliability','double'), ('resolution','double'), 
                ('uncertainty','double'), ('crps_potential','double')])

    return crps_decompos, reliab_table

def alpha(yobs, ysim, pp_cst = 0.3):
    ''' Score computing the Pvalue of the Kolmogorov-Smirnov test '''

    pit = [hypit(o,s,pp_cst) 
                    for o,s in zip(yobs,ysim)] 
    nval = len(pit)
    uniform_dens = (np.arange(nval)+1-pp_cst)/(nval+1-2*pp_cst)
    distances = np.sort(pit)-uniform_dens
    max_dist = np.max(np.abs(distances))

    return kolmogorov(np.sqrt(nval)*max_dist)

def iqr_scores(obs, ens):
    ''' Compute the interquartile range (iqr) divided by clim and iqr reliability'''

    iqr_clim = sutils.percentiles(obs, [25., 75.]) 
    
    nforc = ens.shape[0]
    assert len(obs)==nforc

    nens = ens.shape[1]
    iqr = np.zeros((nforc,3))
    rel = np.zeros(nforc)

    for i in range(nforc):
        iqr[i, :2] = sutils.percentiles(ens[i,:], [25., 75.])
        iqr[i, 2] = iqr[i,1]-iqr[i,0]
        rel[i] = int( (obs[i]>=iqr[i,0]) & (obs[i]<=iqr[i,1]) )

    pre_sc = float(np.mean(iqr[:,2])/np.diff(iqr_clim))
    rel_sc = float(np.mean(rel) * 100)
        
    out = {'precision_skill': 1-pre_sc,
             'reliability_skill': 1-abs(rel_sc-50.)/50., 
             'precision_score': pre_sc,
             'reliability_score': rel_sc}
    return out, iqr, iqr_clim, rel

def median_contingency(obs, ens):
    ''' Compute the contingency matrix for below/above median forecast.
        A positive event is equivalent to below median (i.e. dry)

        Contigency matrix is presented as follows:
        ----------------------------------------------
        | obs<med & ens<med   | obs<med & ens>=med   |
        |--------------------------------------------|
        | obs>=med & ens<med  | obs>=med & ens>=med  |
        ----------------------------------------------
    '''

    obs_med = sutils.percentiles(obs, 50.).values[0]
    
    nforc = ens.shape[0]
    assert len(obs)==nforc

    cont = np.zeros((2,2))
    for i in range(nforc):
        med_obs = int(obs[i]>= obs_med)
        umed = np.mean(ens[i,:]>= obs_med)
        cont[med_obs, int(round(umed))] += 1

    hit = (cont[0,0] + cont[1,1] +0.)/np.sum(cont)
    miss_low = (0.+cont[1,0])/np.sum(cont[:,0])

    return cont, hit, miss_low, obs_med

def tercile_contingency(obs, ens):
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
    obs_t1 = sutils.percentiles(obs, 100./3).values[0]
    obs_t2 = sutils.percentiles(obs, 200./3).values[0]
    
    nforc = ens.shape[0]
    assert len(obs)==nforc

    cont = np.zeros((3,3))
    for i in range(nforc):
        t_obs = (obs[i]>= obs_t1).astype(int) + (obs[i]>=obs_t2).astype(int)
        uu = (ens[i,:] >= obs_t1).astype(int) + (ens[i,:] >= obs_t2).astype(int)
        ut = np.bincount(uu)
        cont[t_obs, np.argmax(ut)] += 1

    hit = (cont[0,0] + cont[1,1] + cont[2,2] + 0.)/np.sum(cont)
    hit_low = (cont[0,0] + 0.)/np.sum(cont[:,0]) 
    hit_high = (cont[2,2] + 0.)/np.sum(cont[:,2])
    miss_low = (0.+np.sum(cont[1:,0]))/np.sum(cont[:,0])

    return cont, hit, miss_low, hit_low, hit_high, obs_t1, obs_t2

def det_metrics(yobs,ysim, compute_persistence=False, min_val=0., eps=1):
    """
        Compute deterministic performance metrics
        
        :param np.array yobs: Observations
        :param np.array ysim: Simulated data
        :param bool compute_persistence: Compute persistence metrics
                (will restrict data valid data)
        :param float min_val: Threshold below which data is considered missing
        :param float eps: Value added to 0 when computing log

    """

    # inputs
    assert len(yobs)==len(ysim)
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
   
    persist = np.nan
    persist_inv = np.nan
    if compute_persistence:
        persist = 1.0 - np.mean(np.square(e))/np.mean(np.square(esh))
        persist_inv = 1.0 - np.mean(np.square(einv))/np.mean(np.square(esh_inv))

    metrics  = {'nse':nse, 'nselog':nselog, 'nseinv':nseinv,
            'persist':persist, 'persist_inv':persist_inv, 
            'nseinv':nseinv, 'bias':bias, 
            'biaslog':biaslog, 'biasinv':biasinv,
            'absbias':abs(bias), 
            'absbiaslog':abs(biaslog), 'absbiasinv':abs(biasinv),
            'corr':corr, 'ratiovar':ratiovar}

    return metrics, idx


def ens_metrics(yobs,ysim, pp_cst=0.3, min_val=0.):
    """
        Computes a set of ensemble performance metrics
        
        :param np.array yobs: Observations (n values)
        :param np.array ysim: Simulated ensemble data (nxp values)
        :param float pp_cst:  Constant used to compute plotting position
        :param float min_val: Threshold below which data is considered missing

    """

    # inputs
    yobs = np.array(yobs, copy=False)
    ysim = np.array(ysim, copy=False)
    assert len(yobs)==ysim.shape[0]

    # find proper data
    idx = np.isfinite(yobs)
      
    # alpha score
    al = alpha(yobs[idx], ysim[idx,:])

    # crps
    cr, rt = crps(yobs[idx],ysim[idx,:])

    # iqr
    iqr = iqr_scores(yobs[idx], ysim[idx,:])

    # contingency tables
    cont_med, hit_med, miss_medlow, obs_med = median_contingency(yobs[idx], ysim[idx,:])
    cont_terc, hit_terc, miss_terclow, hit_terclow, hit_terchigh, obs_t1, obs_t2 = tercile_contingency(yobs[idx], ysim[idx,:])

    # FCVF skill scores
    rmse_fcvf = np.repeat(np.nan, 3)
    rmsep_fcvf = rmse_fcvf
    crps_fcvf = rmse_fcvf
    if HAS_FCVF:
        crps_fcvf = skillscore.crps(yobs[idx],ysim[idx,:],yobs[idx])
        rmse_fcvf = skillscore.rmse(yobs[idx],ysim[idx,:],yobs[idx])
        rmsep_fcvf = skillscore.rmsep(yobs[idx],ysim[idx,:],yobs[idx])

    metrics = {'alpha': al,
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
            'crps': cr['crps'][0],
            'crps_potential': cr['crps_potential'][0],
            'crps_uncertainty': cr['uncertainty'][0],
            'crps_reliability': cr['reliability'][0],
            'crps_skill': (1-cr['crps'][0]/cr['uncertainty'][0])*100,
            'crps_potential_skill' : (1-cr['crps_potential'][0]/cr['uncertainty'][0])*100,
            'crps_reliability_skill' : (1-cr['reliability']/cr['uncertainty'][0])*100,
            'crps_skill_fcvf': crps_fcvf[0],
            'rmse_skill_fcvf': rmse_fcvf[0],
            'rmsep_skill_fcvf': rmsep_fcvf[0],
            'crps_score_fcvf': crps_fcvf[1],
            'rmse_score_fcvf': rmse_fcvf[1],
            'rmsep_score_fcvf': rmsep_fcvf[1],
            'crps_ref_fcvf': crps_fcvf[2],
            'rmse_ref_fcvf': rmse_fcvf[2],
            'rmsep_ref_fcvf': rmsep_fcvf[2]}

    return metrics, idx, rt, cont_med, cont_terc


def ens_metrics_names():
    
    o = np.random.lognormal(size=10)
    s = np.random.normal(size=(10,20))
    sc, idx, rt, cont_med, cont_terc = ens_metrics(o, s)
    
    return sc.keys()

def det_metrics_names():
    
    o = np.random.lognormal(size=10)
    s = np.random.normal(size=(10,1))
    sc, idx = det_metrics(o, s)
    
    return sc.keys()
