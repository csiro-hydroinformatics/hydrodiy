from pathlib import Path
import math

import numpy as np
import pandas as pd

from scipy.stats import kstest, percentileofscore, spearmanr

import warnings

from hydrodiy.stat import transform
from hydrodiy.io import csv

from hydrodiy import has_c_module
if has_c_module("stat", False):
    import c_hydrodiy_stat

EPS = 1e-10

# Reads Cramer-Von Mises table
CVPATH = Path(__file__).resolve().parent / "data" /\
    "cramer_von_mises_test_pvalues.zip"
CVM_TABLE, _ = csv.read_csv(CVPATH, index_col=0)
CVM_NSAMPLE = CVM_TABLE.columns.values.astype(int)
CVM_QQ = CVM_TABLE.index.values
CVM_TABLE = CVM_TABLE.values


def __check_ensemble_data(obs, ens):
    """ Check dimensions of obs and ens data """

    # Convert data to proper dimensions
    obs = np.atleast_1d(obs).astype(np.float64)
    if obs.ndim > 1:
        obs = obs.squeeze()
    if obs.ndim > 1:
        raise ValueError("obs is not 1D")

    ens = np.atleast_2d(ens).astype(np.float64)

    # Check dimensions
    if ens.shape[0] != obs.shape[0]:
        errmess = "Expected ens with first dim equal to"\
            + f"{obs.shape[0]}, got {ens.shape[0]}."
        raise ValueError(errmess)

    # Skip nan
    idx = pd.notnull(obs)
    idx = idx & pd.notnull(ens).any(axis=1)
    if np.sum(idx) == 0:
        raise ValueError("No valid data")
    obs = obs[idx]
    ens = ens[idx, :]

    # Dimensions
    nforc = obs.shape[0]
    nens = ens.shape[1]

    return obs, ens, nforc, nens


def __nonulldata(tobs, tsim):
    """ Exclude nan data from obs and sim """

    idx = pd.notnull(tobs) & pd.notnull(tsim)
    if np.sum(idx) == 0:
        raise ValueError("No valid data in transformed space")

    nok = np.sum(idx)
    nval = len(tobs)
    if nok != nval:
        warnmess = f"There are {nval-nok} null values in"\
                   + "transformed data."
        warnings.warn(warnmess)

    return tobs[idx], tsim[idx]


def pit(obs, ens, random=False, cst=0.3, kind="rank", censor=0.):
    """
    Compute probability integral transformed (PIT) values
    for ensemble forecasts.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array
    random : bool
        Randomize forecast and obs to generate pseudo-pit
    cst : float
        Constant used to compute plotting positions if random is True
    kind : str
        Argument passed to percentileofscore function from scipy
    censor : float
        Censoring threshold to compute sudo pit (i.e. when obs <= censor
        and ens has members <= censor)

    Returns
    -----------
    pits : numpy.ndarray
        PIT value
    is_sudo : numpy.ndarray
         Tells if the pit is a sudo value
    """
    # Check data
    cst = min(0.5, cst)
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Check sudo pits
    is_sudo = np.zeros(nforc).astype(bool)
    idx = (obs < censor+EPS) & (np.sum(ens < censor + EPS, axis=1) > 0)
    is_sudo[idx] = True

    # Compute pits
    if random:
        dobs = np.random.uniform(-EPS, EPS, size=nforc)
        dens = np.random.uniform(-EPS, EPS, size=(nforc, nens))

        pits = (ens+dens-(obs+dobs)[:, None] < 0).astype(int)
        pits = (np.sum(pits, 1)+0.5-cst)/(1.-cst+nens)
    else:
        pits = np.array([percentileofscore(ensval, obsval, kind)/100.
                        for ensval, obsval in zip(ens, obs)])

    return pits, is_sudo


def crps(obs, ens):
    """ Compute the CRPS decomposition from Hersbach (2000)

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        ensemble forecast data, [n,p] array

    Returns
    -----------
    decompos : pandas.Series
        CRPS decomposition as per Equations 35 to 39 in Hersbach, 2000
    table : pandas.DataFrame
        Decomposition table, as per
        - Equation 29 for a and b (alpha and beta)
        - Equation 30 for g
        - Equation 36 for reliability
    """
    has_c_module("stat")

    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # set weights to zero and switch off use of weights
    weights = np.zeros(nforc, dtype=np.float64)
    use_weights = 0

    # run C code via cython
    table = np.zeros((nens+1, 7), dtype=np.float64)
    decompos = np.zeros(5, dtype=np.float64)
    is_sorted = 0
    ierr = c_hydrodiy_stat.crps(use_weights, is_sorted, obs, ens,
                                weights, table, decompos)
    if ierr != 0:
        raise ValueError(f"c_crps returns {ierr}")

    table = pd.DataFrame(table)
    table.columns = ["freq", "a", "b", "g", "rank",
                     "reliability", "crps_potential"]

    index = ["crps", "reliability", "resolution",
             "uncertainty", "potential"]
    decompos = pd.Series(decompos, index=index)

    return decompos, table


def anderson_darling_test(unifdata):
    """ Compute the Anderson Darling (AD) test statistic for a
    uniformly distributed variable and its pvalue using the code
    provided by Marsaglia and Marsaglia (2004):
    Marsaglia, G., & Marsaglia, J. (2004).
    Evaluating the Anderson-Darling Distribution. Journal of Statistical
    Software, 9(2), 1 - 5. doi:http://dx.doi.org/10.18637/jss.v009.i02

    Parameters
    -----------
    unifdata : numpy.ndarray
        1d data vector in [0, 1]

    Returns
    -----------
    pvalue : float
        AD test pvalue
    adstat : float
        AD test statistic
    """
    has_c_module("stat")

    # Check data
    unifdata = np.atleast_1d(unifdata).astype(np.float64)

    # set function outputs
    outputs = np.zeros(2, dtype=np.float64)

    # run C code via cython
    ierr = c_hydrodiy_stat.ad_test(unifdata, outputs)

    if ierr != 0:
        raise ValueError(f"ad_test returns {ierr}")

    adstat = outputs[0]
    pvalue = outputs[1]

    return adstat, pvalue


def cramer_von_mises_test(data):
    """ Perform the Cramer-Von Mises test using a uniform
    distribution as a null hypothesis. The pvalue are computed
    from table prepared with the R package goftest (function pCvM)

    Parameters
    -----------
    data : numpy.ndarray
        1d data vector

    Returns
    -----------
    cvstat : float
        CV test statistic
    pvalue : float
        CV test pvalue
    """

    # Compute Cramer-Von Mises statistic
    nsample = data.shape[0]
    unif = (2*np.arange(1, nsample+1)-1).astype(float)/2/nsample
    cvstat = 1./12/nsample + np.sum((unif-np.sort(data))**2)

    # Find closest sample population
    idx = np.argmin(np.abs(nsample-CVM_NSAMPLE))
    cdf = CVM_TABLE[:, idx]

    # Interpolate pvalue
    pvalue = np.interp(cvstat, CVM_QQ, cdf)

    return cvstat, pvalue


def alpha(obs, ens, cst=0.3, type="CV", sudo_perc_threshold=5):
    """ Score computing the Pvalue of the Cramer Von-Mises test (CV) or
    Kolmogorov-Smirnov test (KS)

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        simulated data, [n,p] array
    cst : float
        Constant used in the computation of the plotting position
    type : str
        Type of alpha score. CV is Cramer Von-Mises, KS is Kolmogorov-Smirnov,
        AD is Anderson Darling.
    sudo_perc_threshold : float
        Percentage threshold for warning about too many sudo pits

    Returns
    -----------
    stat : float
        Test statistic (low values mean that the test is passed)
    pvalue : float
        Test pvalue (values close to one mean that the test is passed)
    is_sudo : numpy.ndarray
         Tells if the pit is a sudo value.
         See hydrodiy.stat.metrics.pit
    """
    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    # Compute pit
    pits, is_sudo = pit(obs, ens, random=True)

    # Warning if too much sudo pits
    if np.sum(is_sudo) > nforc*float(sudo_perc_threshold)/100:
        warnings.warn(f"More than {sudo_perc_threshold}% sudo" +
                      " pits in pits series.")

    if type == "KS":
        # KS test
        stat, pvalue = kstest(pits, "uniform")
    elif type == "CV":
        # Cramer Von-Mises test
        stat, pvalue = cramer_von_mises_test(pits)
    elif type == "AD":
        # Anderson Darlin test
        stat, pvalue = anderson_darling_test(pits)
    else:
        raise ValueError("Expected test type in [CV/KS/AD]," +
                         f" got {type}")

    return stat, pvalue, is_sudo


def iqr(ens, ref, coverage=50.):
    """ Compute the interquantile range skill score (iqr)

    Parameters
    -----------
    ens : numpy.ndarray
        simulated data, [n,p] array
    ref : numpy.ndarray
        climatology data, [n,p] array
    coverage : float
        Interval coverage. For example, if coverage=50,
        the score will be computed between 25% and 75% percentiles

    Returns
    -----------
    skill : float
        IQR skill score computed as
        1/n Sum (clim[i]-score[i])/(clim[i]+score[i])

        Interpretation of this scores is:
        - a value close to 100 indicates perfect IQR, or that the forecast is
          close to a deterministic ensemble
        - a value close to 0 indicates same IQR than climatology
        - a value close to -100 indicates that climatologyis is close
          to a deterministic ensemble

    score : float
        IQR score

    clim : float
        IQR score of climatology

    ratio : float
        IQR ratio computed as
        1/n Sum score[i]/clim[i]

        Interpretation of this scores is:
        - a value close to 0 indicates perfect IQR, or that the forecast is
          close to a deterministic ensemble
        - a value close to 1 indicates same IQR than climatology
        - a value close to +infinity indicates that climatology is close
          to a deterministic ensemble
    """

    # Check data
    ens = np.atleast_2d(ens)
    ref = np.atleast_2d(ref)
    nforc, _ = ens.shape

    if ref.shape[0] != nforc:
        raise ValueError(f"Expected clim to have {ref.shape[0]} forecasts, "
                         + f"got {ens.shape[0]}")

    # Initialise
    iqr = np.zeros((nforc, 3))
    iqr_clim = np.zeros((nforc, 3))

    # Coverage percentiles
    perc = [coverage/2, 100.-coverage/2]

    # Loop through forecasts
    for i in range(nforc):
        iqr_clim[i, :2] = np.nanpercentile(ref[i, :], perc)
        iqr_clim[i, 2] = iqr_clim[i, 1] - iqr_clim[i, 0]

        iqr[i, :2] = np.nanpercentile(ens[i, :], perc)
        iqr[i, 2] = iqr[i, 1] - iqr[i, 0]

    skill = 100*np.mean((iqr_clim[:, 2]-iqr[:, 2])/(iqr_clim[:, 2]+iqr[:, 2]))
    ratio = np.mean(iqr[:, 2]/iqr_clim[:, 2])
    score = np.mean(iqr[:, 2])
    clim = np.mean(iqr_clim[:, 2])

    return skill, score, clim, ratio


def bias(obs, sim, trans=transform.Identity(), excludenull=False,
         type="standard"):
    """ Compute simulation bias

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    transform : hydrodiy.stat.transform.Transform
        Data transforma object
    excludenull : bool
        Exclude pair of data where obs or sim are nan or inf
    type : str
        Type of bias computed.
        standard = 1-s/o
        normalised = (s-o)/(s+o)
        log = log(s)-log(o)

    Returns
    -----------
    bias_value : float
        Simulation bias
    """
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape != sim.shape:
        raise ValueError("Expected sim with dim equal " +
                         f"to {obs.shape}, got{sim.shape}")

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    if excludenull:
        tobs, tsim = __nonulldata(tobs, tsim)

    # Compute mean(obs) and mean(sim)
    meano = np.mean(tobs)
    if abs(meano) < EPS:
        warnings.warn("Mean value of obs is close to " +
                      f"zero ({meano:3.3e}), returning nan")
        return np.nan

    means = np.mean(tsim)

    # Compute bias depending on type
    if type == "standard":
        bias_value = (means-meano)/meano
    elif type == "normalised":
        bias_value = (means-meano)/(means+meano)
    elif type == "log":
        if means > EPS and meano > EPS:
            bias_value = math.log(means)-math.log(meano)
        else:
            warnings.warn("Cannot compute bias-log with " +
                          f"mo = {meano:0.2f} and ms = {means:0.2f}.")
            return np.nan
    else:
        raise ValueError("Expected type in " +
                         f"[standard/normalised/log], got {type}.")

    return bias_value


def nse(obs, sim, trans=transform.Identity(), excludenull=False):
    """ Compute Nash-Sucliffe efficiency.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
    trans : hydrodiy.stat.transform.Transform
        Data transform object
    excludenull : bool
        Exclude pair of data where obs or sim are nan or inf

    Returns
    -----------
    nse_value : float
        Nash-Sutcliffe efficiency (N)

    """
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape[0] != sim.shape[0]:
        raise ValueError("Expected sim with dim equal " +
                         f"to {obs.shape[0]}, got {sim.shape[0]}.")

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)

    if excludenull:
        tobs, tsim = __nonulldata(tobs, tsim)

    # SSE
    errs = np.sum((tsim-tobs)**2)

    mo = np.mean(tobs)
    erro = np.sum((mo-tobs)**2)

    nse_value = 1-errs/erro

    return nse_value


def dscore(obs, sim, eps=1e-6):
    """ Compute the discrimination score (D score) for continuous
    forecasts as per

    Weigel, Andreas P., and Simon J. Mason.
    "The generalized discrimination score for ensemble forecasts."
    Monthly Weather Review 139.9 (2011): 3069-3074.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
    eps : float
        Tolerance to detect ties

    Returns
    -----------
    D : float
        D score value:
        * D=0.0 means that the model as an inverse discrimination (i.e.
                forecasting high when obs is low)
        * D=0.5 means that the model is not discriminating
        * D=1.0 means that the model is perfectly discriminating

    """
    has_c_module("stat")

    # Check data
    obs = np.atleast_1d(obs).astype(np.float64)
    sim = np.atleast_2d(sim).astype(np.float64)
    eps = np.float64(eps)

    if sim.ndim != 2:
        raise ValueError("Expected sim of dimension 2, " +
                         f"got {sim.shape}.")

    nval, nens = sim.shape

    if nens == 1:
        # Compute ensemble rank for deterministic forecasts
        franks = np.argsort(np.argsort(sim[:, 0]))
    else:
        # initialise data
        fmat = np.zeros((nval, nval), dtype=np.float64)
        franks = np.zeros(nval, dtype=np.float64)

        # Compute ensemble rank for ensemble forecasts
        c_hydrodiy_stat.ensrank(eps, sim, fmat, franks)

    # Compute obs rank
    oranks = np.argsort(np.argsort(obs))

    # Compute rank correlation
    D = (np.corrcoef(oranks, franks)[0, 1]+1)/2

    return D


def kge(obs, sim, trans=transform.Identity(), excludenull=False):
    """ Compute Kling-Gupta efficiency index.

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n], [n,1], or [n,p] array
    trans : hydrodiy.stat.transform.Transform
        Data transform object
    excludenull : bool
        Exclude pair of data where obs or sim are nan or inf

    Returns
    -----------
    kge_value : float
        KGE index

    """
    # Check data
    obs = np.atleast_1d(obs)
    sim = np.atleast_1d(sim)

    if obs.shape[0] != sim.shape[0]:
        raise ValueError("KGE - Expected sim with dim equal " +
                         f"to {obs.shape[0]}, got {sim.shape[0]}.")

    # Transform
    tobs = trans.forward(obs)
    tsim = trans.forward(sim)
    if excludenull:
        tobs, tsim = __nonulldata(tobs, tsim)

    # Means
    meano = np.mean(tobs)
    if abs(meano) < EPS:
        warnings.warn("KGE - Mean value of obs is close to " +
                      f"zero ({meano:3.3e}), returning nan")
        return np.nan

    means = np.mean(tsim)

    # Standard deviations
    stdo = np.std(tobs)
    stds = np.std(tsim)

    if abs(stdo) < EPS:
        warnings.warn("KGE - Standard dev of obs is close to " +
                      f"zero ({stdo:3.3e}), returning nan.")
        return np.nan

    # Correlation
    if abs(stds) > EPS:
        corr = np.corrcoef(tobs, tsim)[0, 1]
    else:
        warnings.warn("KGE - Standard dev of sim is close to " +
                      f"zero ({stds:3.3e}), cannot compute correlation, " +
                      "returning nan")
        return np.nan

    # KGE
    kge_value = 1-math.sqrt((1-means/meano)**2+(1-stds/stdo)**2+(1-corr)**2)

    return kge_value


def corr(obs, ens, trans=transform.Identity(),
         excludenull=False, stat="median", type="Pearson", censor=1e-10):
    """ Compute correlation coefficient

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    ens : numpy.ndarray
        simulated data, [n,p] array
    trans : hydrodiy.stat.transform.Transform
        Data transform object
    excludenull : bool
        Exclude pair of data where obs or sim are nan or inf
    stat : str
        Use median or mean from ensemble
    type : str
        Pearson, Spearman or censored type of correlation
    censor : float
        Censoring threshold (used only if type is "censored")

    Returns
    -----------
    corr_value : float
        Correlation coefficient
    """
    # Convert ens to 2d
    ens = np.atleast_2d(ens)
    if ens.shape[0] == 1:
        ens = ens.T

    # Check data
    obs, ens, nforc, nens = __check_ensemble_data(obs, ens)

    if stat not in ["median", "mean"]:
        raise ValueError("Expected stat in [mean/median], got "+stat)

    if type not in ["Pearson", "Spearman", "censored"]:
        raise ValueError("Expected type in [Pearson/Spearman/censored]," +
                         f" got {type}.")

    # Transform
    tobs = trans.forward(obs)
    tens = trans.forward(ens)

    # Compute statistic
    if stat == "mean":
        tsim = np.nanmean(tens, axis=1)
    else:
        tsim = np.nanmedian(tens, axis=1)

    if excludenull:
        tobs, tsim = __nonulldata(tobs, tsim)

    # Check std
    stdo = np.std(tobs)
    if abs(stdo) < EPS:
        warnings.warn("CORR - Standard dev of obs is close to " +
                      f"zero ({stdo:3.3e}), returning nan")
        return np.nan

    # Compute
    if type == "Pearson":
        corr_value = np.corrcoef(tobs, tsim)[0, 1]
    else:
        corr_value = spearmanr(tobs, tsim).correlation

    return corr_value


def absolute_peak_error(obs, sim, winerase=300, winpeakbefore=5,
                        winpeakafter=10, neventmax=500):
    """ Mean absolute peak time error

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    winerase : int
        Number of time steps before and after the peak that is deleted
        after processing a peak event.
    winpeakbefore : int
        Number of time step before peak to define event window.
    winpeakafter : int
        Number of time step after peak to define event window.

    Returns
    -----------
    aperr : float
        Average peak timing error
    events : pandas.DataFrame
        Characteristics of each events.
    """

    # Check data
    obsc = np.array(obs).squeeze().copy()
    obsc[obsc < 0] = np.nan

    simc = np.array(sim).squeeze().copy()
    simc[simc < 0] = np.nan

    # Initialise
    nval = len(obsc)
    nvalid = nval
    nevent = 0
    aperr = 0

    # Run
    events = []
    while nvalid > winerase and nevent < neventmax:
        # find max obs
        imax = np.nanargmax(obsc)
        idx = np.clip(np.arange(imax-winpeakbefore,
                      imax+winpeakafter+1), 0, nval-1)

        # find peak location in both obs and sim
        imaxo = np.nanargmax(obsc[idx])
        imaxs = np.nanargmax(simc[idx])
        delta = abs(imaxs-imaxo)
        aperr += delta
        events.append({"start": idx[0], "end": idx[-1],
                       "delta": delta, "imax_obs": imaxo,
                       "imax_sim": imaxs})

        # loop
        idx = np.clip(np.arange(imax-winerase, imax+winerase+1),
                      0, nval-1)
        obsc[idx] = np.nan
        nvalid -= len(idx)
        nevent += 1

    events = pd.DataFrame(events)

    return aperr/nevent, events


def relative_percentile_error(obs, sim, percentile_range,
                              eps=0., modified=False, neval=50):
    """ Mean relative percentile error

    Parameters
    -----------
    obs : numpy.ndarray
        obs data, [n] or [n,1] array
    sim : numpy.ndarray
        simulated data, [n] or [n,1] array
    percentile_range : list
        Range of percentiles over which relative bias is computed.
    eps : float
        Small term added to denominator in relative bias computation.
    modified : bool
        Compute a modified relative bias as:
        B = (qo-qs)/(qo+qs)
        instead of the standard relative bias:
        B = (qo-qs)/(eps+qo)

        Note that if modified is True, eps is not used.
    neval : int
        Number of percentile computed between rg[0] and rg[1].

    Returns
    -----------
    aperr : float
        Average peak timing error
    events : pandas.DataFrame
        Characteristics of each events.
    """
    # Check values
    rg = percentile_range
    if len(rg) != 2:
        raise ValueError("Expected len(rg) = 2, got {0}".format(len(rg)))
    if rg[1] < rg[0]+1:
        raise ValueError(f"Expected rg[1] > rg[0]+1, got rg={rg[0]}, {rg[1]}.")

    for ir, r in enumerate(rg):
        if r < 0 or r > 100:
            raise ValueError(f"Expected r[{ir}] in [0, 100], got {r}.")

    # Process
    idx = pd.notnull(obs) & pd.notnull(sim)
    rperr = 0.
    perc = []
    for iq, qt in enumerate(np.linspace(*percentile_range, num=neval)):
        qo = np.percentile(obs[idx], qt)
        qs = np.percentile(sim[idx], qt)
        if modified:
            rp = (qs-qo)/(qo+qs) if not np.isclose(qo+qs, 0.) else 0.
        else:
            rp = (qs-qo)/(qo+eps)

        rperr += abs(rp)
        perc.append({"quantile": qt, "qobs": qo, "qsim": qs,
                     "rel_perc_err": rp})

    rperr /= neval
    perc = pd.DataFrame(perc)

    return rperr, perc


def confusion_matrix(obs, sim, ncat=None):
    """ Compute confusion matrix from binary forecats

    Parameters
    -----------
    obs : np.array
        True/False observed
    sim : np.array
        True/False forecast
    ncat : int
        Number of expected categories. If None will
        infer from categories present in obs and sim.

    Returns
    -----------
     conf_mat : np.array
        Confusion matrix. For binary forecasts, it is organised
        as follows:
            | True Negatives,  False Positives |
            | False Negatives, True Positives  |
    """
    # Check inputs
    obs = np.array(obs).astype(np.int32)
    sim = np.array(sim).astype(np.int32)

    # First pass using pandas
    cm = pd.crosstab(obs, sim)

    # Infer number of categories
    if ncat is None:
        cats = np.concatenate([cm.index.values, cm.columns.values])
        ncat = len(np.unique(cats))

    # Add missing rows and columns
    if cm.shape != (ncat, ncat):
        for icat in range(ncat):
            if icat not in cm.columns:
                cm.loc[:, icat] = 0
            if icat not in cm.index:
                cm.loc[icat, :] = 0

        # Re-order
        cm = cm.loc[:, np.arange(ncat)]
        cm = cm.loc[np.arange(ncat), :]

    return cm


def binary(conf_mat):
    """ Metrics computed from binary forecasts
    See https://en.wikipedia.org/wiki/Confusion_matrix and
    Stephenson (2000) and Stephenson et al. (2008) for the
    definition of scores.

    Stephenson, David B. "Use of the odds ratio for diagnosing
    forecast skill." Weather and Forecasting 15.2 (2000): 221-232.

    Stephenson, David B., et al. "The extreme dependency score: a
    non-vanishing measure for forecasts of rare events."
    Meteorological Applications 15.1 (2008): 41-50.

    Parameters
    -----------
    conf_mat : np.array
        Confusion matrix organised as follows (caution this may be different
        from litterature):
            | True Negatives,  False Positives |
            | False Negatives, True Positives  |

        See hydrodiy.stats.metrics.confusion_matrix

    Returns
    -----------
    scores : dict
        The binary metrics computed ire as follows:
        * truepos: Number of cases with concurrent sim and obs
        * falsepos: Number of cases with sim, but no-obs
        * trueneg: Number of cases with concurrent no-sim and no-obs
        * falseneg: Number of cases with no-sim, but obs
        * bias : Proportion of number of forecast events over number of obs
                            events
        * hit_rate: Proportion of correct forecasts given obs happened
                     This is also called sensitivity or recall.
        * precision: Proportion of correct forecasts given sim happened
        * false_alarm: Proportion of incorrect forecasts given obs happened
                     This is also called miss rate or false negative rate.
        * accuracy: Proportion of correct predictions (i.e. truepos+trueneg)
                    compared to total number of events. This is also
                    called proportion correct.
        * F1: F score. This is the harmonic mean of hit rate and precision.
        * MCC: Matthews correlation coefficient. This is identical to
               the square root of the normalised Chi-squared measures
               of association (also referred to as "Phi").
        * LOR: Log odds ratio.
        * ORSS: Odd ratio skill score
        * EDS: Extreme dependency score (see Stephenson et al., 2008)

    scores_rand : dict
        Scores computed when forecast and obs are considered independent.
        The following scores are computed: accuracy, hitrate, falsealarm,
        precision, MCC, oddsratio, F1.
    """
    # Process confusion matrix
    conf_mat = np.array(conf_mat, dtype=np.int64)
    if conf_mat.shape != (2, 2):
        raise ValueError("Expected confusion matrix of shape " +
                         f"(2, 2), got {conf_mat.shape}.")
    # TP: true positive (hit)
    # FP: false positive (false alarm)
    # FN: false negative (miss)
    # TN: true negative (hit)
    ((TN, FP), (FN, TP)) = conf_mat

    Pobs = TP+FN
    Nobs = TN+FP
    Psim = TP+FP
    Nsim = TN+FN
    nval = Pobs+Nobs

    # Compute scores
    H = TP/Pobs
    F = FP/Nobs
    theta = H*(1-F)/(1-H)/F

    LOR = np.nan
    if H > 0 and H < 1 and F > 0 and F < 1:
        LOR = math.log(theta)

    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    EDS = np.nan
    if TP > 0:
        EDS = 2*math.log(Pobs/nval)/math.log(TP/nval)-1

    ORSS = np.nan
    if theta > -1 and theta < 1:
        ORSS = (theta-1)/(theta+1)

    # Random values
    TP_rand = Pobs*Psim/nval
    TN_rand = Nobs*Nsim/nval
    FP_rand = Nobs*Psim/nval
    FN_rand = Pobs*Nsim/nval

    # Random scores
    H_rand = TP_rand/Pobs
    F_rand = FP_rand/Nobs
    theta_rand = H_rand*(1-F_rand)/(1-H_rand)/F_rand
    MCC_rand = (TP_rand*TN_rand-FP_rand*FN_rand)
    MCC_rand /= math.sqrt((TP_rand + FP_rand) * (TP_rand + FN_rand)
                          * (TN_rand + FP_rand) * (TN_rand + FN_rand))

    # Generate scores
    scores = {
        "truepos": TP, "falsepos": FP,
        "trueneg": TN, "falseneg": FN,
        "bias": Psim/Pobs,
        "hitrate": H,
        "precision": TP/Psim,
        "falsealarm": F,
        "accuracy": (TP+TN)/nval,
        "F1":  2*TP/(2*TP+FP+FN),
        "MCC": MCC,
        "LOR": LOR,
        "ORSS": ORSS,
        "EDS": EDS
    }

    scores_rand = {
        "accuracy": (TP_rand+TN_rand)/nval,
        "hitrate": TP_rand/(TP_rand+FN_rand),
        "falsealarm": FP_rand/(FP_rand+TN_rand),
        "precision": TP_rand/(TP_rand+FP_rand),
        "MCC": MCC_rand,
        "LOR": math.log(theta_rand),
        "F1":  2*TP_rand/(2*TP_rand+FP_rand+FN_rand)
    }

    return scores, scores_rand
