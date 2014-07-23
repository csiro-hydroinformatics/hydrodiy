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

def iqr_scores(obs, ens, coverage=80):
    ''' Compute the interquantile range (iqr) divided by clim and iqr reliability'''

    iqr_clim = [np.percentile(obs,q) 
                    for q in [(100-coverage)/2, (100+coverage)/2]]
    
    nforc = ens.shape[0]
    assert len(obs)==nforc

    nens = ens.shape[1]
    iqr = np.zeros((nforc,3))
    rel = np.zeros(nforc)

    for i in range(nforc):
        iqr[i, :2] = [np.percentile(ens[i,:],q) 
                            for q in [(100-coverage)/2, (100+coverage)/2]]
        iqr[i, 2] = iqr[i,1]-iqr[i,0]
        rel[i] = int( (obs[i]>=iqr[i,0]) & (obs[i]<=iqr[i,1]) )
        
    return {'precision': 1-np.mean(iqr[:,2])/np.diff(iqr_clim), 
             'reliability': 100*np.mean(rel)/coverage - 1, 
             'iqr':iqr, 
             'iqr_clim':iqr_clim, 
             'rel':rel}

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

    e = yobs[idx] - ysim[idx]
    elog = np.log(eps+yobs[idx]) - np.log(eps+ysim[idx])
    einv = 1/(eps+yobs[idx]) - 1/(eps+ysim[idx])
    if compute_persistence:
        esh = yobs[idx] - yobs_shift[idx]
        esh_inv = 1/(eps+yobs[idx]) - 1/(eps+yobs_shift[idx])
    
    # Obseved mean and variance
    mo = np.mean(yobs[idx])
    vo = np.var(yobs[idx])
    vlogo = np.var(np.log(eps+yobs[idx]))
    vinvo = np.var(1/(eps+yobs[idx]))
    
    # metrics
    nse = 1.0 - np.mean(np.square(e))/vo
    nselog = 1.0 - np.mean(np.square(elog))/vlogo
    nseinv = 1.0 - np.mean(np.square(einv))/vinvo
    corr = np.corrcoef(yobs[idx],ysim[idx],rowvar=1)[0,1]
    bias = np.mean(e)/mo
    ratiovar = np.var(ysim[idx])/vo
   
    persist = np.nan
    persist_inv = np.nan
    if compute_persistence:
        persist = 1.0 - np.mean(np.square(e))/np.mean(np.square(esh))
        persist_inv = 1.0 - np.mean(np.square(einv))/np.mean(np.square(esh_inv))
    metrics  = {'idx':idx, 
            'nse':nse, 'nselog':nselog, 
            'persist':persist, 'persist_inv':persist_inv, 
            'nseinv':nseinv, 'bias':bias, 
            'corr':corr, 'ratiovar':ratiovar}

    return metrics


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
    iqr80 = iqr_scores(yobs[idx], ysim[idx,:], coverage = 80)
    iqr50 = iqr_scores(yobs[idx], ysim[idx,:], coverage = 50)

    # FCVF skill scores
    rmse_fcvf = np.repeat(np.nan, 3)
    rmsep_fcvf = rmse_fcvf
    crps_fcvf = rmse_fcvf
    if HAS_FCVF:
        crps_fcvf = skillscore.crps(yobs[idx],ysim[idx,:],yobs[idx])
        rmse_fcvf = skillscore.rmse(yobs[idx],ysim[idx,:],yobs[idx])
        rmsep_fcvf = skillscore.rmsep(yobs[idx],ysim[idx,:],yobs[idx])

    metrics = {'idx':idx, 
            'alpha': al,
            'iqr80_precision': iqr80['precision'],
            'iqr80_reliability': iqr80['reliability'],
            'iqr50_precision': iqr50['precision'],
            'iqr50_reliabity': iqr50['reliability'],
            'crps': cr['crps'],
            'crps_potential': cr['crps_potential'],
            'crps_uncertainty': cr['uncertainty'],
            'crps_reliability': cr['reliability'],
            'crps_reliability_table': rt,
            'crps_skill_fcvf': crps_fcvf[0],
            'rmse_skill_fcvf': rmse_fcvf[0],
            'rmsep_skill_fcvf': rmsep_fcvf[0],
            'crps_score_fcvf': crps_fcvf[1],
            'rmse_score_fcvf': rmse_fcvf[1],
            'rmsep_score_fcvf': rmsep_fcvf[1],
            'crps_ref_fcvf': crps_fcvf[2],
            'rmse_ref_fcvf': rmse_fcvf[2],
            'rmsep_ref_fcvf': rmsep_fcvf[2]}

    return metrics

