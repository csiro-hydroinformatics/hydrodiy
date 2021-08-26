#
# Code written by Andrew Schepen, CSIRO
#
#
#


import os, sys

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import stats



n_ens = {"sys4_raw":15, "sys4_csiro_dm":15, "clim_ref":200, "ind_cal":200, "qm":15, "mp_cal":200}
model_colours = {"obs":"black", "sys4_raw":"red", "clim_ref":"orange", "ind_cal":"purple"}
model_key = {"obs":"O", "sys4_raw":"A", "clim_ref":"B", "ind_cal":"C"}


def pit(forecast, obs, censor=False):

    n = len(forecast)
    pp = plotting_positions(n, "hazen")

    pit = np.interp(obs, np.sort(forecast), pp, left=0, right=1)
    if obs == 0.0 and censor:
        pit = np.random.uniform(0, pit)

    return pit


def pit_alpha(forecasts, obs, censor=False):

    num_samples = forecasts.shape[1]
    num_events = forecasts.shape[0]

    pp = plotting_positions(num_samples, "hazen")

    pits = []

    for i in range(num_events):
        pit = np.interp(obs[i], np.sort(forecasts[i, :]), pp, left=0, right=1)
        if obs[i] == 0.0 and censor:
            ppit = np.random.uniform(0, pit)
            pits.append(ppit)
        else:
            pits.append(pit)

    sorted_pits = np.sort(pits)
    pp = plotting_positions(num_events, "weibull")

    sumdif = 0
    for i in range(num_events):
        sumdif += abs(sorted_pits[i] - pp[i])

    alpha = 1.0 - (2.0 / float(num_events) * sumdif)

    # Use sorted_pits since the below line requires a numpy array to work (and it"s guaranteed to be a numpy array).
    ksee = 1.0 - (np.sum(sorted_pits == 0) + np.sum(sorted_pits==1))/float(num_events)

    pit_res = {}

    pit_res["pits"] = pits
    pit_res["sorted_pits"] = sorted_pits
    pit_res["rel1_alpha"] = alpha
    pit_res["rel2_ksee"] = ksee

    return pit_res



def crps_ss_old(forecasts, obs, ref_forecasts=None):

    # forecasts.shape = (num_events, num_samples)
    # obs.shape = (num_events)
    # ref_forecasts.shape = (num_events, num_samples)

    num_samples = forecasts.shape[1]
    num_events = forecasts.shape[0]

    scores = crps_bom(forecasts[:,:], obs[:])

    crps_res = {}

    if ref_forecasts is not None:

        ref_scores = crps_bom(ref_forecasts[:, :], obs[:])

        skill_score = 1.0 - np.mean(scores)/np.mean(ref_scores)

        crps_res["skill_score"] = skill_score


    crps_res["mean"] = np.mean(scores)
    crps_res["scores"] = scores

    return crps_res

def crps_ss(forecasts, obs, ref_forecasts=None):

    # forecasts.shape = (num_events, num_samples)
    # obs.shape = (num_events)
    # ref_forecasts.shape = (num_events, num_samples)

    num_samples = forecasts.shape[1]
    num_events = forecasts.shape[0]

    scores  = []
    for i in range(num_events):
        # print(crps_ecdf(forecasts[i,:], obs[i]))
        scores.append(crps_ecdf(forecasts[i,:], obs[i]))

    crps_res = {}

    if ref_forecasts is not None:

        ref_scores = []
        for i in range(num_events):
            # print(crps_ecdf(forecasts[i, :], obs[i]))
            ref_scores.append(crps_ecdf(ref_forecasts[i, :], obs[i]))

        skill_score = 1.0 - np.mean(scores)/np.mean(ref_scores)

        crps_res["skill_score"] = skill_score



    crps_res["mean"] = np.mean(scores)
    crps_res["scores"] = scores

    return crps_res

def bias(forecasts, obs):

    # forecasts.shape = (num_events, num_samples)
    # obs.shape = (num_events)


    num_samples = forecasts.shape[1]
    num_events = forecasts.shape[0]

    fc_means = []
    errors = []
    for i in range(num_events):

        fc_means.append(np.mean(forecasts[i,:]))
        errors.append(fc_means[i]-obs[i])

    bias_res = {}

    bias_res["mean_err"] = np.mean(errors)
    bias_res["mean_abs_err"] = np.mean(np.abs(errors))
    bias_res["rel"] = np.mean(errors)/np.mean(obs)*100.0
    bias_res["fc_means"] = fc_means
    bias_res["errors"] = errors
    bias_res["mult"] = np.mean(forecasts)/np.mean(obs)

    return bias_res


def rmse(forecasts, obs, fc_centre="median", ref_forecasts=None):

    # forecasts.shape = (num_events, num_samples)
    # obs.shape = (num_events)
    # ref_forecasts.shape = (num_events, num_samples)

    assert fc_centre in ["mean", "median"]

    num_samples = forecasts.shape[1]
    num_events = forecasts.shape[0]

    if fc_centre == "median":
        errors = np.median(forecasts[:, :], axis=1) - obs
    else:
        errors = np.mean(forecasts[:, :], axis=1) - obs

    scores = np.sqrt(np.mean(errors * errors))

    rmse_res = {}

    if ref_forecasts is not None:

        ref_scores = []
        for i in range(num_events):
            if fc_centre == "median":
                errors = np.median(ref_forecasts[:, :], axis=1) - obs
            else:
                errors = np.mean(ref_forecasts[:, :], axis=1) - obs

            ref_scores = np.sqrt(np.mean(errors * errors))

        skill_score = 1.0 - np.mean(scores)/np.mean(ref_scores)

        rmse_res["skill_score"] = skill_score

    rmse_res["mean"] = np.mean(scores)
    rmse_res["scores"] = scores

    return rmse_res


def lag1_corr(ens1, ens2):

    # ens1 and ens2 are forecasts 1-day apart
    # if 1D, must have dimensions of (num_events)
    # if 2D, must have dimensions of (num_events, num_ensembles)
    # if 2D, the correlation is calculated for each ensemble member and then averaged

    ndim = len(ens1.shape)
    assert ndim in [1,2]
    assert ens1.shape == ens2.shape

    corr = []

    if ndim == 1:
        # r = stats.spearmanr(ens1, ens2)[0]
        r = stats.kendalltau(ens1, ens2)[0]
        corr.append(r)

    elif ndim == 2:
        num_samples = ens1.shape[1]
        for i in range(num_samples):
            # r = stats.spearmanr(ens1[:, i], ens2[:, i])[0]
            r = stats.kendalltau(ens1[:, i], ens2[:, i])[0]

            corr.append(r)

    corr_res = {}

    # When calculating the mean/median correlation,
    # omit cases where there is no correlation, i.e. one of the variables has no variance,
    # e.g. [0.0, 0.0, 0.0] or [0.5, 0.5, 0.5]
    corr_res["corr"] = np.nanmean(corr)

    return corr_res

def lag1_corr_event(ens1, ens2):

    # ens1 and ens2 are forecasts 1-day apart
    # if 1D, must have dimensions of (num_lead_times)
    # if 2D, must have dimensions of (num_ensembles, num_lead_times)
    # if 2D, the correlation is calculated for each ensemble member and then averaged

    ndim = len(ens1.shape)
    assert ndim in [1,2]
    assert ens1.shape == ens2.shape

    corr = []

    if ndim == 1:
        # r = stats.spearmanr(ens1, ens2)[0]
        r = stats.kendalltau(ens1, ens2)[0]
        corr.append(r)

    elif ndim == 2:
        num_samples = ens1.shape[0]
        for i in range(num_samples):
            # r = stats.spearmanr(ens1[i,:], ens2[i,:])[0]
            r = stats.kendalltau(ens1[i, :], ens2[i, :])[0]

            corr.append(r)

    corr_res = {}

    # When calculating the mean/median correlation,
    # omit cases where there is no correlation, i.e. one of the variables has no variance,
    # e.g. [0.0, 0.0, 0.0] or [0.5, 0.5, 0.5]
    corr_res["corr"] = np.nanmean(corr)

    return corr_res



def interp_probs(points, data, probs):
    """
    Interpolate probabilities of points.
    data should be monotonically increasing.
    """
    # Make sure that data is monotically increasing.
    try:
        assert np.all(data == np.sort(data))
    except AssertionError:
        print(len(data))
        assert np.all(data == np.sort(data))
    # Jitter the data for the case that the data have duplicate values.
    # np.interp() cannot handle duplicate values properly.
    data = data + np.random.uniform(low=-1E-10, high=1E-10, size=len(data))
    # Need to resort it because jittering can change the order of duplicate values.
    data = np.sort(data)

    point_probs = np.interp(points, data, probs, left=0.0, right=1.0)
    point_probs[np.less_equal(points, np.min(data))] = 0.0
    point_probs[np.greater_equal(points, np.max(data))] = 1.0

    return point_probs


def plotting_positions(n, type="hazen"):

    # n = len(data)

    if type.lower() == "hazen":
        # good for normal and gumbel shaped distributions
        p = [(i - 0.5)/float(n) for i in range(1, n + 1)]
    elif type.lower() == "weibull":
        # good for uniform distribution
        p = [i/(n+1.0) for i in range(1, n + 1)]
    elif type.lower() == "cunnane":
        # compromise for all distributions
        p = [(i-0.4)/(n+0.2) for i in range(1, n + 1)]
    else:
        raise ValueError("{} is not a supported plotting position")

    return p

def crps_ecdf(f_ens, o):

    f_ens.sort()

    o_cdf = f_cdf = f_prev = crps = 0
    m = len(f_ens)

    for i in range(m):
        f_n = f_ens[i]
        if o_cdf == 0 and o < f_n:
            crps += (o - f_prev) * f_cdf ** 2
            crps += (f_n - o) * (f_cdf - 1) ** 2
            o_cdf = 1
        else:
            crps += ((f_n - f_prev) * (f_cdf - o_cdf) ** 2)

        f_cdf = (i+1.)/m # ecdf
        f_prev = f_n

    # obs to the right of the forecast.
    if o_cdf == 0:
        crps += o - f_n

    return crps


def crps_bom(ysim, yobs):

    ys_sort = np.sort(ysim, axis=-1)

    rangeobs = np.arange(len(yobs))

    def f(yo, ys, probs, ymin=None, ymax=None):

        nsamples = 500
        if ymin is None:
            ymin = min(np.min(ys), yo)
        if ymax is None:
            ymax = max(np.max(ys), yo)

        if ymin == ymax == yo:
            # prevent NaN
            return 0.0

        delta = (ymax - ymin)/nsamples
        yi  = np.arange(ymin, ymax, delta)

        pvals = interp_probs(yi, ys, probs)
        steps = np.maximum(np.sign(yi - yo), 0.0)
        bins = np.square(pvals - steps) * delta

        return np.sum(bins[np.isfinite(bins)])

    ymin = []
    ymax = []
    for i in rangeobs:

        ymin.append(np.min([np.min(ysim[i]), yobs[i]]))
        ymax.append(np.max([np.max(ysim[i]), yobs[i]]))

    prob_sim = plotting_positions(len(ys_sort[0]), "hazen")
    crps_scores = [f(yobs[i], ys_sort[i], prob_sim, ymin[i], ymax[i]) for i in rangeobs]


    return crps_scores
