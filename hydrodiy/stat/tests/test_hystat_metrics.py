import re, math, sys
from pathlib import Path
import pytest
import warnings
from itertools import product as prod

import time
import zipfile

import matplotlib as mpl
mpl.use("Agg")
print("Using {} backend".format(mpl.get_backend()))

import matplotlib.pyplot as plt

from scipy.stats import norm, lognorm, spearmanr
import numpy as np
import pandas as pd

from hydrodiy import has_c_module
from hydrodiy.stat import metrics
from hydrodiy.io import csv
from hydrodiy.stat import transform, sutils

from hydrodiy import has_c_module
if has_c_module("stat", False):
    import c_hydrodiy_stat

from hydrodiy.plot import putils

from vrf_scores import crps_ecdf as crps_csiro

np.random.seed(0)

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"
FIMG.mkdir(exist_ok=True)

fd1 = FTEST/"data"/"crps_testdata_01.txt"
data = np.loadtxt(fd1)
OBS1 = data[:,0].copy()
SIM1 = data[:,1:].copy()

frt1 = FTEST/"data"/"crps_testres_crpsmatens_01.txt"
CRPS_RELIABTAB1 = np.loadtxt(frt1)

fv1 = FTEST/"data"/"crps_testres_crpsvalens_01.txt"
c1 = np.loadtxt(str(fv1))
c1[2] *= -1
CRPS_VALUE1 = {
    "crps":c1[0],
    "reliability":c1[1],
    "resolution":c1[2],
    "uncertainty":c1[3],
    "potential":c1[4]
}

fd2 = FTEST/"data"/"crps_testdata_02.txt"
data = np.loadtxt(fd2)
OBS2 = data[:,0].copy()
SIM2 = data[:,1:].copy()

frt2 = FTEST/"data"/"crps_testres_crpsmatens_02.txt"
CRPS_RELIABTAB2 = np.loadtxt(frt2)

fv2 = FTEST/"data"/"crps_testres_crpsvalens_02.txt"
c2 = np.loadtxt(fv2)
c2[2] *= -1
CRPS_VALUE2 = {
    "crps":c2[0],
    "reliability":c2[1],
    "resolution":c2[2],
    "uncertainty":c2[3],
    "potential":c2[4]
}

TRANSFORMS = [
    transform.Identity(),
    transform.Log(),
    transform.Reciprocal()
]

hascmodule = pytest.mark.skipif(
    not has_c_module("stat", False), reason="Missing C module c_hydrodiy_stat"
)


iswindows = pytest.mark.skipif(
    re.search("win", sys.platform), reason="Windows os specific error not fixed yet"
)

def test_alpha():
    nval = 100
    nens = 500
    obs = np.linspace(0, 10, nval)
    sim = np.repeat(np.linspace(0, 10, nens)[:, None], \
                nval, 1).T

    a, _, _ = metrics.alpha(obs, sim)
    assert (np.allclose(a, 1.))

@hascmodule
def test_crps_csiro():
    nval = 100
    nens = 1000
    nrepeat = 100
    sparam = 1
    sign = 2

    for irepeat in range(nrepeat):
        obs = lognorm.rvs(size=nval, s=sparam, loc=0, scale=1)
        noise = norm.rvs(size=(nval, nens), scale=sign)
        trend = norm.rvs(size=nval, scale=sign*1.5)
        ens = obs[:, None]+noise+trend[:, None]

        cr, _ = metrics.crps(obs, ens)

        # Reference computation
        ccr = [crps_csiro(forc, o) for forc, o in zip(ens, obs)]
        assert (np.isclose(cr.crps, np.mean(ccr)))


@hascmodule
def test_crps_reliability_table1():
    cr, rt = metrics.crps(OBS1, SIM1)
    for i in range(rt.shape[1]):
        assert (np.allclose(rt.iloc[:, i], \
            CRPS_RELIABTAB1[:,i], atol=1e-5))


@hascmodule
def test_crps_reliability_table2():
    cr, rt = metrics.crps(OBS2, SIM2)
    for i in range(rt.shape[1]):
        assert (np.allclose(rt.iloc[:, i], \
            CRPS_RELIABTAB2[:,i], atol=1e-5))


@hascmodule
def test_crps_value1():
    cr, rt = metrics.crps(OBS1, SIM1)
    for nm in cr.keys():
        ck = np.allclose(cr[nm], CRPS_VALUE1[nm], atol=1e-5)
        assert (ck)


@hascmodule
def test_crps_value2():
    cr, rt = metrics.crps(OBS2, SIM2)
    for nm in cr.keys():
        ck = np.allclose(cr[nm], CRPS_VALUE2[nm], atol=1e-5)
        assert (ck)


def test_pit():
    nforc = 100
    nens = 200
    obs = np.linspace(0, 1, nforc)
    ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)
    pit, sudo = metrics.pit(obs, ens)

    assert (np.all(np.abs(obs-pit)<8e-3))
    assert (np.all(~sudo[1:]))
    assert (sudo[0])


def test_pit_hassan():
    obs = [3]
    ens = [0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]

    # using scipy func
    pit, sudo = metrics.pit(obs, ens, random=False)
    assert (np.isclose(pit, 0.5666666666666))
    assert (np.all(~sudo))

    # using randomisation
    nrepeat = 1000
    pits = np.array([metrics.pit(obs, ens, random=True)[0] \
                for i in range(nrepeat)]).squeeze()
    pits = pd.Series(pits).value_counts().sort_index()
    assert (np.allclose(pits.index, [0.33121, 0.394904, \
                    0.458599, 0.522293, 0.585987, 0.649682, 0.713376]))
    assert ((pits>100).all())


def test_pit_hassan_rpp():
    frpp = FTEST / "data" / "rpp_data.zip"
    with zipfile.ZipFile(frpp, "r") as archive:
        obs, _ = csv.read_csv("obs.csv", archive=archive, index_col=0, \
                                            has_colnames=False)
        ens, _ = csv.read_csv("rpp_ensemble.csv", archive=archive, \
                                        index_col=0)
    pit1, sudo1 = metrics.pit(obs, ens, random=False)
    pit2, sudo2 = metrics.pit(obs, ens, random=True)

    plt.close("all")
    try:
        fig, ax = plt.subplots()
    except:
        pytest.skip("Cannot initialise matplotlib, not too sure why")

    ff = sutils.ppos(len(pit1))

    kk = np.argsort(pit1)
    p1 = pit1[kk]
    s1 = sudo1[kk]
    ax.plot(p1[s1], ff[s1], "k.", \
                markersize=5, alpha=0.3, \
                label="Scipy percentileofscore (sudo)")
    ax.plot(p1[~s1], ff[~s1], "o",
                markeredgecolor="k", \
                markerfacecolor="k", \
                label="Scipy percentileofscore")


    kk = np.argsort(pit2)
    p2 = pit2[kk]
    s2 = sudo2[kk]
    ax.plot(p2[s2], ff[s2], "r.", \
                markersize=5, alpha=0.5, \
                label="hydrodiy.metrics.pit using random=True (sudo)")

    ax.plot(p2[~s2], ff[~s2], "o", \
                markeredgecolor="r", \
                markerfacecolor="r", \
                label="hydrodiy.metrics.pit using random=True")

    ax.legend(loc=2, framealpha=0.5)
    ax.set_xlabel("PIT [-]")
    ax.set_ylabel("ECDF [-]")

    fp = FIMG / "pit_hassan.png"
    fig.savefig(fp)


def test_cramer_von_mises():
    fd = FTEST/"data"/"cramer_von_mises_test_data.csv"
    data = pd.read_csv(fd, skiprows=15).values

    fe = FTEST/"data"/"cramer_von_mises_test_data_expected.csv"
    expected = pd.read_csv(fe, skiprows=15)

    for nval in expected["nval"].unique():
        # Select data for nval
        exp = expected.loc[expected["nval"] == nval, :]

        for i in range(exp.shape[0]):
            x = data[i, :nval]

            st1 = exp["stat1"].iloc[i]
            pv1 = exp["pvalue1"].iloc[i]

            st2, pv2 = metrics.cramer_von_mises_test(x)

            ck1 = abs(st1-st2)<5e-3
            ck2 = abs(pv1-pv2)<1e-2

            assert (ck1 and ck2)


def test_cramer_von_mises2():
    """ Second test of Cramer Von Mises test """

    fd = FTEST / "data" / "testdata_AD_CVM.csv"
    data = pd.read_csv(fd, skiprows=15)
    cc = [cn for cn in data.columns if re.search("^x", cn)]

    for _, row in data.iterrows():
        unifdata = row[cc]
        st1 = row["CVM_stat"]
        pv1 = row["CVM_pvalue"]
        st2, pv2 = metrics.cramer_von_mises_test(unifdata)
        ck = np.allclose([st1, pv1], [st2, pv2], atol=1e-3)
        assert (ck)

@hascmodule
def test_anderson_darling():
    fd = FTEST / "data" / "testdata_AD_CVM.csv"
    data = pd.read_csv(fd, skiprows=15)
    cc = [cn for cn in data.columns if re.search("^x", cn)]

    for _, row in data.iterrows():
        unifdata = row[cc]
        st1 = row["AD_stat"]
        pv1 = row["AD_pvalue"]
        st2, pv2 = metrics.anderson_darling_test(unifdata)

        ck = np.allclose([st1, pv1], [st2, pv2])
        assert (ck)

@hascmodule
def test_anderson_darling_error():
    nval = 20
    unifdata = np.random.uniform(0, 1,  size=nval)
    unifdata[-1] = 10
    msg = "ad_test"
    with pytest.raises(ValueError, match=msg):
        st, pv = metrics.anderson_darling_test(unifdata)


def test_alpha():
    nforc = 100
    nens = 200
    nrepeat = 50

    for i in range(nrepeat):
        obs = np.linspace(0, 1, nforc)
        ens = np.repeat(np.linspace(0, 1, nens)[None, :], nforc, 0)

        for type in ["CV", "KS", "AD"]:
            st, pv, sudo = metrics.alpha(obs, ens)
            assert (pv>1.-1e-3)


def test_iqr():
    """ Testing IQR for normally distributed forecasts """
    nforc = 100
    nens = 200

    ts = np.repeat(np.random.uniform(10, 20, nforc)[:, None], \
                                        nens, 1)
    qq = sutils.ppos(nens)
    spread = 2*norm.ppf(qq)[None,:]
    ens = ts + spread

    # Double the spread. This should lead to iqr skill score of 33%
    # sk = (2*iqr-iqr)/(2*iqr+iqr) = 1./3
    ref = ts + spread*2

    iqr = metrics.iqr(ens, ref)
    expected = np.array([100./3, 2.67607, 5.35215, 0.5])
    assert (np.allclose(iqr, expected, atol=1e-4))


def test_iqr_error():
    """ Testing IQR error """
    ens = np.random.uniform(0, 1, (100, 50))
    ref = np.random.uniform(0, 1, (80, 50))
    msg = "Expected clim"
    with pytest.raises(ValueError, match=msg):
        iqr = metrics.iqr(ens, ref)


def test_bias():
    """ Test bias """
    obs = np.arange(0, 200).astype(float)

    for trans in TRANSFORMS:
        if trans.params.nval > 0:
            trans.params.values[0] = np.mean(obs)*1e-2

        tobs = trans.forward(obs)
        tsim = tobs - 2
        sim = trans.backward(tsim)
        bias = metrics.bias(obs, sim, trans)

        mo = np.mean(tobs)
        expected = -2./mo
        ck = np.isclose(bias, expected)
        assert (ck)

        bias = metrics.bias(obs, sim, trans, type="normalised")
        ms = np.mean(tsim)
        expected = (ms-mo)/(ms+mo)
        ck = np.isclose(bias, expected)
        assert (ck)

        if mo > 0 and ms > 0:
            bias = metrics.bias(obs, sim, trans, type="log")
            expected = math.log(ms)-math.log(mo)
            ck = np.isclose(bias, expected)
            assert (ck)


def test_bias_error():
    """ Test bias error """
    obs = np.arange(0, 200)
    sim = np.arange(0, 190)
    msg = "Expected sim"
    with pytest.raises(ValueError, match=msg):
        bias = metrics.bias(obs, sim)

    sim = np.arange(0, 200)
    msg = "Expected type"
    with pytest.raises(ValueError, match=msg):
        bias = metrics.bias(obs, sim, type="bidule")


def test_nse():
    """ Testing  NSE """
    obs = np.arange(0, 200)+100.
    bias = -1.

    for trans in TRANSFORMS:
        if trans.params.nval > 0:
            trans.params.values[0] = np.mean(obs)*1e-2

        tobs = trans.forward(obs)
        tsim = tobs + bias
        sim = trans.backward(tsim)
        nse = metrics.nse(obs, sim, trans)

        expected = 1-bias**2*len(obs)/np.sum((tobs-np.mean(tobs))**2)
        assert (np.isclose(nse, expected))


def test_nse_error():
    """ Test bias error """
    obs = np.arange(0, 200)
    sim = np.arange(0, 190)
    msg = "Expected sim"
    with pytest.raises(ValueError, match=msg):
        bias = metrics.nse(obs, sim)


@hascmodule
@iswindows
def test_ensrank_weigel_data():
    # Testing ensrank C function  against data from
    # Weigel and Mason (2011)
    sim = np.array([[22, 23, 26, 27, 32], \
        [28, 31, 33, 34, 36], \
        [24, 25, 26, 27, 28]], dtype=np.float64)

    fmat = np.zeros((3, 3), dtype=np.float64)
    ranks = np.zeros(3, dtype=np.float64)
    eps = np.float64(1e-6)

    c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

    fmat_expected = np.array([\
            [0., 0.08, 0.44], \
            [0., 0., 0.98], \
            [0., 0., 0.]])
    assert (np.allclose(fmat, fmat_expected))

    ranks_expected = [1., 3., 2.]
    assert (np.allclose(ranks, ranks_expected))


@hascmodule
def test_ensrank_deterministic():
    nval = 5
    nrepeat = 100
    for i in range(nrepeat):
        sim = np.random.uniform(0, 1, (nval, 1))
        fmat = np.zeros((nval, nval), dtype=np.float64)
        ranks = np.zeros(nval, dtype=np.float64)
        eps = np.float64(1e-6)

        c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

        # Zero on the diagonal
        assert (np.allclose(np.diag(fmat), np.zeros(nval)))

        # Correct rank
        ranks_expected = 1+np.argsort(np.argsort(sim[:, 0]))

        xx, yy = np.meshgrid(sim[:, 0], sim[:, 0])
        tmp = (xx>yy).astype(float).T
        fmat_expected = np.zeros((nval, nval))
        idx = np.triu_indices(nval)
        fmat_expected[idx] = tmp[idx]

        assert (np.allclose(ranks, ranks_expected))
        assert (np.allclose(fmat, fmat_expected))

@hascmodule
@iswindows
def test_ensrank_python():
    nval = 4
    nens = 5
    nrepeat = 100
    eps = np.float64(1e-6)

    for irepeat in range(nrepeat):
        for ties in [True, False]:
            if ties:
                sim = np.round(np.random.uniform(0, 100, (nval, nens))/10)
            else:
                sim = np.random.uniform(0, 100, (nval, nens))

            fmat = np.zeros((nval, nval), dtype=np.float64)
            ranks = np.zeros(nval, dtype=np.float64)
            c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)

            # Run python
            fmat_expected = fmat * 0.
            for i in range(nval):
                for j in range(i+1, nval):
                    # Basic rankings
                    comb = np.concatenate([sim[i], sim[j]])
                    rk = np.argsort(np.argsort(comb))+1.

                    # Handle ties
                    for cu in np.unique(comb):
                        idx = np.abs(comb-cu)<eps
                        if np.sum(idx)>0:
                            rk[idx] = np.mean(rk[idx])

                    # Compute rank sums
                    srk = np.sum(rk[:nens])
                    fm = (srk-(nens+1.)*nens/2)/nens/nens;
                    fmat_expected[i, j] = fm

            # Ranks
            F = fmat_expected.copy()
            idx = np.tril_indices(nval)
            F[idx] = 1.-fmat_expected.T[idx]
            c1 = np.sum((F>0.5).astype(int), axis=1)
            c2 = np.sum(((F>0.5-1e-8) & (F<0.5+1e-8)).astype(int), \
                            axis=1)
            ranks_expected = c1+0.5*c2

            ck = np.allclose(fmat, fmat_expected)
            assert (ck)

            ck = np.allclose(ranks, ranks_expected)
            assert (ck)

@hascmodule
def test_dscore_perfect():
    nval = 10
    nens = 100
    obs = np.arange(nval)

    sim = obs[:, None] + np.random.uniform(-1e-3, 1e-3, size=(nval, nens))
    D = metrics.dscore(obs, sim)
    assert (np.allclose(D, 1.))

    sim = -obs[:, None] + np.random.uniform(-1e-3, 1e-3, \
                                        size=(nval, nens))
    D = metrics.dscore(obs, sim)
    assert (np.allclose(D, 0.))

@hascmodule
def test_ensrank_large_ensemble():
    nval = 50
    nens = 5000
    fmat = np.zeros((nval, nval), dtype=np.float64)
    ranks = np.zeros(nval, dtype=np.float64)
    sim = np.random.uniform(0, 1, (nval, nens))
    eps = np.float64(1e-6)

    t0 = time.time()
    c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)
    t1 = time.time()

    # Max 30 sec to compute this
    assert (t1-t0<30)


@hascmodule
def test_ensrank_long_timeseries():
    nval = 1000
    nens = 100
    fmat = np.zeros((nval, nval), dtype=np.float64)
    ranks = np.zeros(nval, dtype=np.float64)
    sim = np.random.uniform(0, 1, (nval, nens))
    eps = np.float64(1e-6)

    t0 = time.time()
    c_hydrodiy_stat.ensrank(eps, sim, fmat, ranks)
    t1 = time.time()

    # Max 3 sec to compute this
    assert (t1-t0<100)


def test_kge():
    obs = np.arange(0, 200)+100.
    bias = 0.1

    for trans in TRANSFORMS:
        if trans.params.nval > 0:
            trans.params.values[0] = np.mean(obs)*1e-2

        # First trial - multiplicative bias
        tobs = trans.forward(obs)
        tsim = tobs*(1+bias)
        sim = trans.backward(tsim)
        kge = metrics.kge(obs, sim, trans)

        expected = 1-math.sqrt(2)*bias
        assert (np.isclose(kge, expected))

        # Second trial - additive bias
        tsim = tobs - bias
        sim = trans.backward(tsim)
        kge = metrics.kge(obs, sim, trans)

        expected = 1-bias/abs(np.mean(tobs))
        assert (np.isclose(kge, expected))

        # Third trial - random error
        tsim = tobs + 1e-2*np.mean(tobs)*np.random.uniform(-1, 1, \
                                                    size=len(tobs))
        sim = trans.backward(tsim)
        kge = metrics.kge(obs, sim, trans)

        bias = np.mean(tsim)/np.mean(tobs)
        rstd = np.std(tsim)/np.std(tobs)
        corr = np.corrcoef(tobs, tsim)[0, 1]
        expected = 1-math.sqrt((1-bias)**2+(1-rstd)**2+(1-corr)**2)
        assert (np.isclose(kge, expected))


def test_kge_warnings():
    # Catch warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        obs = np.zeros(100)
        sim = np.random.uniform(0, 1, size=100)
        try:
            kge = metrics.kge(obs, sim)
        except Warning as warn:
            assert (str(warn).startswith("KGE - Mean value"))
        else:
            raise ValueError("Problem in error handling")

        obs = np.ones(100)
        sim = np.random.uniform(0, 1, size=100)
        try:
            kge = metrics.kge(obs, sim)
        except Warning as warn:
            assert (str(warn).startswith("KGE - Standard dev"))
        else:
            raise ValueError("Problem in error handling")

        obs = np.random.uniform(0, 1, size=100)
        sim = np.ones(100)
        try:
            kge = metrics.kge(obs, sim)
        except Warning as warn:
            assert (str(warn).startswith("KGE - Standard dev"))
        else:
            raise ValueError("Problem in error handling")


def test_corr2d():
    """ Test correlation for ensemble data """
    nval = 200
    nens = 1000
    obs = np.arange(0, nval).astype(float)

    for trans, type, stat in prod(TRANSFORMS, \
                ["Pearson", "Spearman", "censored"], \
                ["mean", "median"]):

        if trans.params.nval > 0:
            trans.params.values[0] = np.mean(obs)*1e-2

        tobs = trans.forward(obs)
        tens = tobs[:, None] - 2 \
                    + np.random.uniform(-2, 2, size=(nval, nens))
        ens = trans.backward(tens)
        corr = metrics.corr(obs, ens, trans, False, stat, type)

        if stat == "mean":
            tsim = np.nanmean(tens, 1)
        else:
            tsim = np.nanmedian(tens, 1)

        if type == "Pearson":
            expected = np.corrcoef(tobs, tsim)[0, 1]
        else:
            expected = spearmanr(tobs, tsim).correlation

        ck = np.isclose(corr, expected)
        assert (ck)


def test_corr1d():
    nval = 200
    obs = np.arange(0, nval).astype(float)

    for trans, type, stat in prod(TRANSFORMS, \
                ["Pearson", "Spearman"], ["mean", "median"]):

        if trans.params.nval > 0:
            trans.params.values[0] = np.mean(obs)*1e-2

        tobs = trans.forward(obs)
        tens = tobs - 2 \
                    + np.random.uniform(-1, 1, size=nval)
        ens = trans.backward(tens)
        corr = metrics.corr(obs, ens, trans, False, stat, type)

        if type == "Pearson":
            expected = np.corrcoef(tobs, tens)[0, 1]
        elif type == "Spearman":
            expected = spearmanr(tobs, tens).correlation
        else:
            X = np.column_stack([tobs, tens])
            _, _, expected = normcensfit2d(X, censor=1e-10)

        ck = np.isclose(corr, expected)
        assert (ck)


def test_abspeakerror():
    """ Test peak timing error using lagged data """
    nval = 2000
    obs = np.exp(np.random.normal(size=nval))

    lag = 3
    sim = np.append(obs[lag:], [0]*2)
    aperr, events = metrics.absolute_peak_error(obs, sim)

    assert (np.isclose(aperr, lag))
    assert (np.allclose(events.delta, lag))


def test_relpercerror():
    """ Test relative percentile error """
    nval = 2000
    obs = np.exp(np.random.normal(size=nval))
    err = 1.3
    sim = (err+1)*obs

    rperr, perc = metrics.relative_percentile_error(obs, sim, [0, 100])
    assert (np.isclose(rperr, err))
    assert (np.allclose(perc.rel_perc_err, err))

    rperr, perc = metrics.relative_percentile_error(obs, sim, [0, 100], \
                                            modified=True)
    errm = err/(2+err)
    assert (np.isclose(rperr, abs(errm)))
    assert (np.allclose(perc.rel_perc_err, errm))


def test_confusion_matrix():
    """ Test computation of confusion matrix """

    for i in range(10):
        # 2x2 confusion matrix
        o, s = [np.random.choice([0, 1], size=100) for i in range(2)]
        m = metrics.confusion_matrix(o, s)

        o = o.astype(bool)
        s = s.astype(bool)
        mexpected = [[np.sum(~o & ~s), np.sum(~o & s)], \
                    [np.sum(o & ~s), np.sum(o & s)]]

        assert (np.allclose(m, mexpected))

        # 5x5 confusion matrix
        o, s = [np.random.choice(np.arange(5), size=100) \
                                for i in range(2)]
        m = metrics.confusion_matrix(o, s) #, ncat=5)

        mexpected = np.zeros((5, 5))
        for i, j in prod(range(5), range(5)):
            n = np.sum((o == i) & (s == j))
            mexpected[i, j] = n

        assert (np.allclose(m, mexpected))


def test_confusion_matrix_missing():
    """ Test computation of confusion matrix with missing categories """

    o = np.random.choice([0, 1], size=100)
    s = np.ones(100, dtype=int)
    m = metrics.confusion_matrix(o, s)
    assert (np.allclose(m.loc[:, 0], [0, 0]))

    o = np.zeros(100, dtype=int)
    m = metrics.confusion_matrix(o, s)
    assert (np.allclose(m.loc[:, 0], [0, 0]))
    assert (np.allclose(m.loc[1, :], [0, 0]))

    o, s = [np.random.choice([0, 1, 3, 4], size=100) for i in [0, 1]]
    m = metrics.confusion_matrix(o, s, ncat=5)
    assert (np.allclose(m.loc[:, 2], [0]*5))
    assert (np.allclose(m.loc[2, :], [0]*5))


def test_binary():
    # Generating Finley forecasts table
    # using dummy variables
    # see Stephenson, David B. "Use of the odds ratio for diagnosing
    #   forecast skill." Weather and Forecasting 15.2 (2000): 221-232.
    # Re-organised confusion matrix to match the function definition
    mat = [[2680, 72], [23, 28]]
    scores, scores_rand = metrics.binary(mat)

    # tests
    assert (scores["truepos"]== 28)
    assert (scores["trueneg"]== 2680)
    assert (scores["falsepos"]== 72)
    assert (scores["falseneg"]== 23)

    # See Table 5 in Stephenson, 2000
    assert (np.isclose(scores["bias"], 1.96, rtol=0, atol=1e-3))
    assert (np.isclose(scores["hitrate"], 0.549, rtol=0, atol=1e-3))
    assert (np.isclose(scores["falsealarm"], 0.026, rtol=0, atol=1e-3))
    assert (np.isclose(scores["LOR"], math.log(45.314), rtol=0, atol=1e-3))
    # .. not exactly the value reported by Stepenson due to rounding
    assert (np.isclose(scores_rand["hitrate"], 0.035, rtol=0, atol=1e-3))
    assert (np.isclose(scores_rand["falsealarm"], 0.036, rtol=0, atol=1e-3))

    # See Table 7 in Stephenson, 2000
    #.. corresponds to the square root of the Pearson
    assert (np.isclose(scores["MCC"], math.sqrt(0.142), rtol=0, atol=1e-3))
    assert (np.isclose(scores_rand["MCC"], 0.0, rtol=0, atol=1e-3))
    assert (np.isclose(scores["accuracy"], 0.966, rtol=0, atol=1e-3))
    assert (np.isclose(scores_rand["accuracy"], 0.948, rtol=0, atol=1e-3))

    # Additional scores
    assert (np.isclose(scores["precision"], 0.28, rtol=0, atol=1e-3))
    assert (np.isclose(scores["EDS"], 0.7396, rtol=0, atol=1e-3))


