import itertools
from pathlib import Path
import pytest
import math
import numpy as np
from scipy import linalg

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import norm

from hydrodiy.data.containers import Vector
from hydrodiy.stat import transform

np.random.seed(0)

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"
FIMG.mkdir(exist_ok=True)

XX = np.exp(np.linspace(-8, 5, 10))


def test_get_transform():
    for nm in transform.__all__:
        # Get transform
        trans = transform.get_transform(nm, mininu=0.5)

        if hasattr(trans, "nu"):
            trans.nu = 0.
        if hasattr(trans, "lam"):
            trans.lam = 0.2
        if hasattr(trans, "xmax"):
            trans.xmax = 1.

        nparams = trans.params.nval
        trans.params.values = np.sort(np.random.uniform(1, 2, nparams))

        x = np.linspace(0, 1, 100)
        x /= 1.5*np.sum(x)
        y = trans.forward(x)

        assert (nm == trans.name)


def test_get_transform_errors():
    """ Test errors in get transform """
    msg = "Expected transform name"
    with pytest.raises(ValueError, match=msg):
        trans = transform.get_transform("bidule")

    # This should work
    trans = transform.get_transform("BoxCox2", lam=1)

    # Test error when no value for constants
    trans = transform.get_transform("BoxCox1lam", lam=0.5)
    msg = "nu is nan"
    with pytest.raises(ValueError, match=msg):
        trans.forward(1.)

    trans = transform.get_transform("BoxCox1nu", nu=0.5)
    msg = "lam is nan"
    with pytest.raises(ValueError, match=msg):
        trans.forward(1.)

    trans = transform.get_transform("LogSinh", loga=-2, logb=0.)
    msg = "xmax is nan"
    with pytest.raises(ValueError, match=msg):
        trans.forward(1.)

    trans = transform.get_transform("Manly", lam=-1)
    msg = "xmax is nan"
    with pytest.raises(ValueError, match=msg):
        trans.forward(1.)


def test_transform_class():
    """ Test the class transform """

    params = Vector(["a"], [0.5], [0.], [1.])
    constants = Vector(["C1", "C2"])
    trans = transform.Transform("test", params, constants)

    assert (trans.name == "test")
    assert (trans.params.names == ["a"])

    # test set and get parameters
    value = 0.5
    trans.params.values = value
    exp = np.array([value], dtype=np.float64)
    assert (np.allclose(trans.params.values, exp))

    # test set and get parameters
    trans.params.values = -1
    exp = np.array([trans.params.mins[0]], dtype=np.float64)
    assert (np.allclose(trans.params.values, exp))

    trans.params.values = 2
    exp = np.array([trans.params.maxs[0]], dtype=np.float64)
    assert (np.allclose(trans.params.values, exp))

    # test setitem/getitem for individual parameters
    trans.a = 0.5
    assert (np.allclose(trans.a, 0.5))
    assert (np.allclose(trans.params.values, [0.5]))
    assert (np.allclose(trans.params["a"], 0.5))

    trans.params["a"] = 0.5
    assert (np.allclose(trans.a, 0.5))
    assert (np.allclose(trans.params.values, [0.5]))
    assert (np.allclose(trans.params["a"], 0.5))

    trans.a = 2.
    trans.b = 2.
    assert (np.allclose(trans.params.values, trans.params.maxs))

    trans.a = -1
    trans.b = -1
    assert (np.allclose(trans.params.values, trans.params.mins))

    # test setitem/getitem for individual constants
    trans.C1 = 6
    assert (np.allclose(trans.C1, 6))
    assert (np.allclose(trans.constants.values, [6, 0.]))
    assert (np.allclose(trans.constants["C1"], 6))

    trans.constants["C1"] = 7
    assert (np.allclose(trans.C1, 7))
    assert (np.allclose(trans.constants.values, [7, 0.0]))
    assert (np.allclose(trans.constants["C1"], 7))


def test_transform_class_error():
    """ Test the error from class transform """

    params = Vector(["a"], [0.5], [0.], [1.])
    trans = transform.Transform("test", params)

    msg = "The truth value of"
    with pytest.raises(ValueError, match=msg):
        trans.a = [10, 10]

    msg = "Cannot set"
    with pytest.raises(ValueError, match=msg):
        trans.a = np.nan

    msg = "Cannot process"
    with pytest.raises(ValueError, match=msg):
        trans.params.values = [np.nan] * trans.params.nval


def test_transform_class_not_implemented():
    # Test the error generation for not implemented methods
    params = Vector(["a"], [0.5], [0.], [1.])
    trans = transform.Transform("test", params)

    # Test not implemented methods
    x = np.linspace(0, 1, 10)
    msg = "Method _forward"
    with pytest.raises(NotImplementedError, match=msg):
        trans.forward(x)

    msg = "Method _backward"
    with pytest.raises(NotImplementedError, match=msg):
        trans.backward(x)

    msg = "Method _jacobian"
    with pytest.raises(NotImplementedError, match=msg):
        trans.jacobian(x)

    msg = "Names are not unique"
    with pytest.raises(ValueError, match=msg):
        params = Vector(["a", "a"], [0.5]*2, [0.]*2, [1.]*2)
        trans = transform.Transform("test", params)


def test_print():
    for nm in transform.__all__:
        trans = transform.get_transform(nm)
        nparams = trans.params.nval
        trans.params.values = np.random.uniform(-5, 5, size=nparams)
        str(trans)


def test_forward_backward():
    for nm in transform.__all__:
        trans = transform.get_transform(nm)
        nparams = trans.params.nval

        if hasattr(trans, "nu"):
            trans.nu = 0.
        if hasattr(trans, "lam"):
            trans.lam = 0.2

        for sample in range(500):
            # generate x sample
            x = np.random.normal(size=100, loc=5, scale=20)
            # Generate parameters
            trans.params.values = np.random.uniform(1e-3, 2, size=nparams)

            # Handle specific cases
            if nm == "Log":
                x = np.exp(x)
            elif nm == "Logit":
                x = np.random.uniform(0., 1., size=100)
                trans.reset()
            elif nm == "LogSinh":
                trans.params.values = [np.random.uniform(-2, -0.1), \
                                            np.random.uniform(-0.1, 0.1)]
            elif nm == "Softmax":
                x = np.random.uniform(0, 1, size=(100, 5))
                x = x/(0.1+np.sum(x, axis=1)[:, None])

            elif nm == "Manly":
                trans.params.values = np.random.uniform(-5, -5)

            if hasattr(trans, "xmax"):
                trans.xmax = x.max()

            # Check x -> forward(x) -> backward(y) is stable
            y = trans.forward(x)
            xx = trans.backward(y)

            # Check raw and transform/backtransform are equal
            if x.ndim>1:
                idx = np.all(~np.isnan(xx) & ~np.isnan(x), axis=xx.ndim-1)
            else:
                idx = ~np.isnan(xx) & ~np.isnan(x)

            ck = np.allclose(x[idx], xx[idx])
            if not ck:
                print(("\n\n!!! Transform {0} failing"+\
                    " the forward/backward test").format(trans.name))

            assert (ck)


def test_jacobian():
    # Test of transformation Jacobian is equal to its
    #    first order numerical approx
    delta = 1e-5

    for nm in transform.__all__:
        trans = transform.get_transform(nm)
        nparams = trans.params.nval

        if hasattr(trans, "nu"):
            trans.nu = 0.
        if hasattr(trans, "lam"):
            trans.lam = 0.2

        for sample in range(100):
            x = np.random.normal(size=100, loc=5, scale=20)
            trans.params.values = np.random.uniform(1e-3, 2, size=nparams)

            if nm in ["Log", "BoxCox2", "BoxCox1nu"]:
                x = np.clip(x, 1e-1, np.inf)

            elif nm == "Logit":
                x = np.random.uniform(1e-2, 1.-1e-2, size=100)
                trans.reset()

            elif nm == "Softmax":
                x = np.random.uniform(0, 1, size=(100, 5))
                x = x/(0.1+np.sum(x, axis=1)[:, None])
                trans.reset()

            elif nm == "LogSinh":
                trans.params.values = [np.random.uniform(-2, -0.1), \
                                            np.random.uniform(-0.1, 0.1)]

            if hasattr(trans, "xmax"):
                trans.xmax = x.max()

            # Compute first order approx of jac
            y = trans.forward(x)

            if nm == "Softmax":
                # A bit more work to compute matrix determinant
                jacn = np.zeros(x.shape[0])
                for i in range(x.shape[1]):
                    xd = x[i, :][None, :] + np.eye(x.shape[1])*delta
                    yd = trans.forward(xd)
                    M = (yd-y[i, :][None, :])/delta
                    jacn[i] = linalg.det(M)
            else:
                yp = trans.forward(x+delta)
                jacn = np.abs(yp-y)/delta

            jac = trans.jacobian(x)

            # Check jacobian are positive
            idx = ~np.isnan(jac)
            ck = np.all(jac[idx]>0.)
            if not ck:
                print(("\n\n!!!Transform {0} not having"+\
                    " a strictly positive Jacobian").format(trans.name))
            assert (ck)

            # Check jacobian is equal to numerical derivation
            idx = idx & (jac>0.) & (jac<5e3)
            crit = np.abs(jac-jacn)/(1+jac+jacn)
            idx = idx & ~np.isnan(crit)
            ck = np.all(crit[idx]<5e-3)
            if not ck:
                print(f"\n\n!!!Transform {trans.name} failing the"+\
                    " numerical Jacobian test "+\
                    f"({crit[idx].max():3.3e}).")

            assert (ck)


def test_transform_censored():
    # Test censored forward transform

    for nm in transform.__all__:
        if nm in ["Softmax"]:
            continue

        trans = transform.get_transform(nm)

        if hasattr(trans, "nu"):
            trans.nu = 0.
        if hasattr(trans, "lam"):
            trans.lam = 0.2

        if hasattr(trans, "xmax"):
            trans.xmax = 1.

        for censor in [0.1, 0.5]:
            tcensor = trans.forward(censor)
            xt = np.linspace(tcensor-1, tcensor+1, 100)
            x = trans.backward(xt)

            xc = trans.backward_censored(xt, censor)

            idx = x>censor
            ck = np.allclose(xc[(~idx) & ~np.isnan(xc)], censor)
            assert (ck)

            ck = np.allclose(xc[idx], x[idx])
            assert (ck)


def test_transform_plot():
    """ Plot transform """
    x = np.linspace(-3, 10, 200)

    xs = np.linspace(0, 0.99, 200)
    xs = np.column_stack([xs, (1-xs)*xs])

    for nm in transform.__all__:
        trans = transform.get_transform(nm)
        nparams = trans.params.nval

        if hasattr(trans, "nu"):
            trans.nu = 0.
        if hasattr(trans, "lam"):
            trans.lam = 0.2
        if hasattr(trans, "xmax"):
            trans.xmax = x.max()

        xx = x
        if nm == "Softmax":
            xx = xs

        plt.close("all")

        try:
            fig, ax = plt.subplots()
        except:
            pytest.skip("Cannot initialise matplotlib, not too sure why")

        for pp in [-20., 0, 20.]:
            trans.reset()
            trans.params.values = trans.params.defaults * (1.+pp/100)
            y = trans.forward(xx)

            xp, yp = xx, y
            if nm == "Softmax":
                xp, yp = xx[:, 0], y[:, 0]

            ax.plot(xp, yp, \
                label="params = default {0}% (rp={1})".format(pp,
                        trans.params))
        ax.legend(loc=4)
        ax.set_title(nm)
        fig.savefig(FIMG/f"transform_{nm}.png")


def test_params_sample():
    nsamples = 1000
    for nm in transform.__all__:
        if nm == "Identity":
            continue

        trans = transform.get_transform(nm)
        samples = trans.params_sample(nsamples)
        assert samples.shape == (nsamples, trans.params.nval)

        mins = trans.params.mins
        assert (np.all(samples>=mins[None, :]-1e-20))

        maxs = trans.params.maxs
        assert (np.all(samples<=maxs[None, :]))

        # Check that all parameters are valid
        for smp in samples:
            trans.params.values = smp


def test_mininu():
    mininu = -10.
    for nm in ["Log", "BoxCox2",  "BoxCox1nu", "Reciprocal"]:
        trans = transform.get_transform(nm, mininu=mininu)
        trans.nu = mininu+1
        assert (np.isclose(trans.nu, mininu+1))

        trans.nu = mininu-1
        assert (np.isclose(trans.nu, mininu))

        if hasattr(trans, "lam"):
            trans.lam = 0.2
        if hasattr(trans, "xmax"):
            trans.xmax = -mininu+1

        # Generate censored inputs
        # ... raw inputs
        nval = 100
        trans.nu = mininu
        x = np.linspace(-mininu-1, -mininu+1, nval)
        y = trans.forward(x)
        # ... fit censored normal dist
        ff = (np.arange(nval)+0.7)/(nval+0.4)
        idx = ~np.isnan(y)
        M = np.column_stack([np.ones(np.sum(idx)), norm.ppf(ff[idx])])
        theta, _, _, _ = linalg.lstsq(M, y[idx])
        y[~idx] = theta[0] + theta[1]*norm.ppf(ff[~idx])
        # ... back transform
        x0 = trans.backward_censored(y)

        # ... test no nan
        assert (np.all(~np.isnan(x0)))


def test_params_logprior():
    nsamples = 1000
    for nm in transform.__all__:
        if nm == "Identity":
            continue

        trans = transform.get_transform(nm)
        samples = trans.params_sample(nsamples)

        for smp in samples:
            trans.params.values = smp
            lp = trans.params_logprior()
            if nm != "LogSinh":
                assert (np.isclose(lp, 0.))
            else:
                assert (np.isclose(lp, norm.logpdf(smp[1], 0, 0.3)))


def test_log_transform():
    """ Test special options for log transform """

    x = np.linspace(1, 3, 100)
    trans = transform.get_transform("Log", nu=0., base=math.exp(1))
    tx = trans.forward(x)
    assert (np.allclose(tx, np.log(x)))

    trans = transform.get_transform("Log", nu=0., base=10)
    tx = trans.forward(x)
    assert (np.allclose(tx, np.log10(x)))


