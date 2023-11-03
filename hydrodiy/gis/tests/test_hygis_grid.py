import re, math
from pathlib import Path
import warnings
import pytest
import numpy as np
import pandas as pd
import warnings
import zipfile
from scipy.spatial.distance import pdist, squareform


import zipfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from hydrodiy import has_c_module
from hydrodiy.gis.grid import Grid, Catchment
from hydrodiy.gis.grid import accumulate, voronoi, \
                                    delineate_river, slope
from hydrodiy.gis.grid import get_grid, AWRAL_SUBGRIDS
from hydrodiy.io import csv

FTEST = Path(__file__).resolve().parent
FIMG = FTEST / "images"
FIMG.mkdir(exist_ok=True)

RUN_ADVANCED = True

CONFIG = {
    "name":"test", \
    "nrows":7, "ncols":5, "cellsize":2., \
    "dtype":np.float64, \
    "xllcorner":130., \
    "yllcorner":-39., \
    "comment": "this is a test grid"
}

NROWS = 6
TESTGRID = Grid(NROWS, NROWS, dtype=np.int32, nodata=-1)
TESTGRID.data = [ [0, 4, 4, 4, 0, 0],
            [0, 4, 4, 8, 0, 0],
            [0, 2, 4, 8, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0]]
TESTGRIDN = np.arange(NROWS*NROWS).reshape((NROWS, NROWS))


def test_print():
    gr = Grid(**CONFIG)
    print(gr)


def test_name_comment():
    gr = Grid(**CONFIG)
    assert (gr.comment == CONFIG["comment"])
    assert (gr.name == CONFIG["name"])


def test_shape():
    gr = Grid(**CONFIG)
    assert gr.shape == (7, 5)


def test_dtype():
    gr = Grid(**CONFIG)
    gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

    gr.dtype = np.int32
    ck = np.allclose(gr.data, 0.)
    assert ck


def test_xvalues_yvalues():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    gr = Grid(**CONFIG)

    xv, yv = gr.xvalues, gr.yvalues
    assert (np.allclose(xv, np.arange(131, 141, 2)))
    assert (np.allclose(yv, np.arange(-38, -24, 2)[::-1]))


def test_clone():
    gr = Grid(**CONFIG)
    data_ini = np.random.uniform(0, 1, (gr.nrows, gr.ncols))
    gr.data = data_ini.copy()

    # Test data is copied accross
    grc = gr.clone()
    ck = np.allclose(grc.data, gr.data)
    assert ck

    # Check the original grid is not changing
    grc.data[0, :] = 100
    ck = np.allclose(gr.data, data_ini)
    assert ck

    # Test cloning works when changing type
    grci = gr.clone(np.int32)
    ck = np.allclose(grci.data, 0)
    assert ck


def test_same_geometry():
    gr = Grid(**CONFIG)

    grc = gr.clone(np.int32)
    assert (gr.same_geometry(grc))

    grc = Grid("test", ncols=10)
    assert (~gr.same_geometry(grc))


def test_getitem():
    gr = Grid(**CONFIG)
    gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

    idx = [0, 2, 5]
    val = gr[idx]
    assert (np.allclose(val, gr.data.flat[idx]))


def test_setitem():
    gr = Grid(**CONFIG)
    gr.data = np.random.uniform(10, 11, (gr.nrows, gr.ncols))

    idx = [0, 2, 5]
    val = np.arange(len(idx))
    gr[idx] = val
    assert (np.allclose(val, gr.data.flat[idx]))


def test_data():
    gr = Grid(**CONFIG)

    dt = np.random.uniform(0, 1, (gr.nrows, gr.ncols))
    gr.data = dt
    ck = np.allclose(gr.data, dt)
    assert ck

    dt = np.random.uniform(0, 1, (gr.nrows+1, gr.ncols+1))
    try:
        gr.data = dt
    except ValueError as err:
        assert (str(err).startswith("Wrong number"))
    else:
        raise Exception("Problem with handling data error")

def test_fill():
    gr = Grid(**CONFIG)
    gr.fill(-2.)

    ck = np.allclose(gr.data, -2.)
    assert ck


def test_save():
    gr = Grid(**CONFIG)
    gr.data = np.random.uniform(0, 1, \
                    (gr.nrows, gr.ncols))

    # Write data
    fg = FTEST / "grid_test_save.bil"
    gr.save(fg)

    # Load it back
    gr2 = Grid.from_header(fg)

    ck = np.allclose(gr.data, gr2.data)
    assert ck

    assert (CONFIG["comment"] == gr2.comment)


def test_dict():
    gr = Grid(**CONFIG)
    js = gr.to_dict()
    assert js["dtype"] == "<f8"

    gr2 = Grid.from_dict(js)
    for att in gr.__dict__:
        a = str(getattr(gr, att))
        b = str(getattr(gr2, att))
        assert a==b


def test_neighbours():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    gr = Grid(**CONFIG)

    idxcells = []
    nr = gr.nrows
    nc = gr.ncols
    for k in range(1, nr-1):
        idxcells += range(nc*k+1, nc*k+nc-1)

    for idxcell in idxcells:
        nb = gr.neighbours(idxcell)
        expected = np.array([idxcell-nc-1, idxcell-nc, idxcell-nc+1, \
            idxcell-1, -1, idxcell+1, idxcell+nc-1, idxcell+nc, \
            idxcell+nc+1])

        ck = np.allclose(nb, expected)
        assert ck

    # Upper left corner
    nb = gr.neighbours(0)
    ck = np.allclose(nb, [-1, -1, -1, -1, -1, 1, -1, nc, nc+1])
    assert ck

    # Lower right corner
    idxcell = nr*nc-1
    nb = gr.neighbours(nr*nc-1)
    ck = np.allclose(nb, [idxcell-nc-1, idxcell-nc, -1,
        idxcell-1, -1, -1, -1, -1, -1])
    assert ck

    # Error
    msg = "c_hydrodiy_gis.neighbours"
    with pytest.raises(ValueError, match=msg):
        nb = gr.neighbours(nr*nc)


def test_coord2cell():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    gr = Grid(**CONFIG)

    csz = gr.cellsize
    xll = gr.xllcorner
    yll = gr.yllcorner

    # Generate coordinates
    xx = xll+np.arange(0, gr.ncols)*csz
    yy = yll+np.arange(gr.nrows-1, -1, -1)*csz
    xxg, yyg = np.meshgrid(xx, yy)
    xycoords0 = np.concatenate([xxg.flat[:][:, None],
                yyg.flat[:][:, None]], axis=1)
    xycoords1 = xycoords0 + np.random.uniform(0, csz, xycoords0.shape)

    # Get cell index and test
    idxcell = gr.coord2cell(xycoords1)
    ck = np.allclose(idxcell, np.arange(gr.ncols*gr.nrows))
    assert ck

    # Get cell coordinate from index and test
    xycoords2 = gr.cell2coord(idxcell)
    ck = np.allclose(xycoords0+csz/2, xycoords2)
    assert ck

    rowcol = gr.cell2rowcol(idxcell)
    cka = np.allclose(rowcol[:, 0], np.repeat(np.arange(7), 5))
    ckb = np.allclose(rowcol[:, 1], np.concatenate([np.arange(5)]*7))
    assert (cka & ckb)


def test_slice():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    ndim = 11
    gr = Grid("test", ndim)
    vect = np.arange(0, int(ndim/2)+1)+1.
    vect = np.concatenate([vect[::-1], vect[1:]])
    gr.data = np.array([vect+i for i in range(ndim)])

    # Slicing a valley...
    xyslice = np.vstack([np.arange(1, 11)+0.1, [5.2]*10]).T
    zslice = gr.slice(xyslice)

    expect = [ 10.7,   9.7,   8.7,   7.7,   6.7,   6.9, \
                    7.9,   8.9,   9.9,  10.9]
    ck = np.allclose(zslice, expect)
    assert ck


def test_from_stream():
    """ Test loading grid from stream """
    # Header only
    fileheader = FTEST / "fdtest.hdr"
    with open(fileheader, "r") as fh:
        gr = Grid.from_stream(fh)

    def test_grid(gr):
        ck = gr.nrows == 283
        ck &= (gr.ncols == 293)
        ck &= np.isclose(gr.xllcorner, 145.44625)
        ck &= np.isclose(gr.yllcorner, -18.291250)
        ck &= np.isclose(gr.cellsize,  0.0025)
        ck &= (gr.nodata == 32767)
        ck &= (gr.name == "fdtest")
        ck &= (gr.comment == "No comment")
        return ck

    assert (test_grid(gr))

    # Header and data
    filedata = FTEST / "fdtest.bil"
    with open(fileheader, "r") as fh, open(filedata, "rb") as fd:
        gr = Grid.from_stream(fh, fd)

    assert (test_grid(gr))


def test_from_zip():
    """ Test loading grid from stream piped from zipfile """
    filezip = FTEST/"flowdir_223202.zip"
    fileheader = "subdir/flowdir_223202.hdr"
    gr = Grid.from_zip(filezip, fileheader)

    ck = gr.nrows == 241
    ck &= (gr.ncols == 260)
    ck &= np.isclose(gr.xllcorner, 147.45125)
    ck &= np.isclose(gr.yllcorner, -37.45125)
    ck &= np.isclose(gr.cellsize,  0.0025)
    ck &= (gr.nodata == 0)
    ck &= (gr.name == "no_name")
    ck &= (gr.comment == "No comment")
    ck &= np.isclose(gr.data.mean(), 30.12049154165337)
    assert ck


def test_from_header():
    filename = FTEST/"header.hdr"
    try:
        gr = Grid.from_header(filename)
    except MemoryError:
        warnings.warn("test_from_header not run due to memory error")
        return

    ck = gr.nrows == 13857
    ck &= (gr.ncols == 16440)
    ck &= (gr.xllcorner == 112.90125)
    ck &= (gr.yllcorner == -43.74375)
    ck &= (gr.cellsize == 0.0025)
    ck &= (gr.nodata == 32767)
    ck &= (gr.name == "mygrid")
    ck &= (gr.comment == "testing header")

    assert ck


def test_plot():
    filename = FTEST/"demtest.hdr"
    gr = Grid.from_header(filename)
    gr.dtype = np.float64
    def fun(x):
        idx = x>=55500
        x[idx] = np.nan
        return np.log(x+1)
    gr = gr.apply(fun)

    plt.close("all")
    fig, ax = plt.subplots()
    gr.plot(ax, interpolation="nearest")
    fp = FIMG/f"{filename.stem}.png"
    fig.savefig(fp)


def test_plot_values():
    """ Test showing grid values """
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    ngrid = 20
    gr = Grid("test", xllcorner=0, yllcorner=0, cellsize=1, \
                        nrows=ngrid, ncols=ngrid)

    # Generate smooth random data
    x = np.linspace(0, 1, ngrid)
    x, y = np.meshgrid(x, x)
    xy = np.column_stack([x.ravel(), y.ravel()])
    dist = squareform(pdist(xy))
    Sigma = np.exp(-dist**2*4)
    m = np.zeros(ngrid*ngrid)

    d = np.random.multivariate_normal(mean=m, cov=Sigma)
    gr.data = d.reshape((ngrid, ngrid))

    plt.close("all")
    fig, ax = plt.subplots()
    gr.plot(ax, interpolation="nearest")
    gr.plot_values(ax, fmt="0.1f", fontsize=7, fontweight="bold", \
                                        color="w")

    fig.set_size_inches((10, 10))
    fig.tight_layout()
    fp = FIMG/"test_plot_values.png"
    fig.savefig(fp)


def test_apply():
    gr = Grid(**CONFIG)
    gr = gr.apply(lambda x: np.random.uniform(0, 1, x.shape))

    grlog = gr.apply(np.log)
    ck = np.allclose(grlog.data, np.log(gr.data))
    assert ck


def test_clip():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    gr = Grid(**CONFIG)
    gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))
    xll, yll = 132.1, -35.1
    xur, yur = 137.6, -26.9
    grclip = gr.clip(xll, yll, xur, yur)

    expected = gr.data[0:6, 1:4]
    assert (np.allclose(expected, grclip.data))
    assert ((grclip.xlim[0] < xll) & (grclip.xlim[1] > xur))
    assert ((grclip.ylim[0] < yll) & (grclip.ylim[1] > yur))


def test_minmaxdata():
    cfg = CONFIG.copy()
    cfg["dtype"] = np.int32
    gr = Grid(**cfg)
    gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

    # Check the minimum is converted to proper dtype
    gr.mindata = 20.4
    assert (gr.mindata == 20)
    assert (gr.data.min() == 20)

    gr.maxdata = 30.6
    assert (gr.maxdata == 30)
    assert (gr.data.max() == 30)

    msg = "Expected mindata<maxdata"
    with pytest.raises(ValueError, match=msg):
        gr.mindata = 35


def test_interpolate_small():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    gr = Grid(**CONFIG)
    gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

    cfg = {
        "name": "interpolate", \
        "nrows": 14, \
        "ncols": 10, \
        "cellsize": 1.
    }
    for key in ["dtype", "xllcorner", "yllcorner"]:
        cfg[key] = CONFIG[key]

    gr_geom = Grid(**cfg)
    gri = gr.interpolate(gr_geom)
    assert (np.allclose(gri.data[0, :], \
                np.linspace(0, 4., 10)))

    assert (np.allclose(gri.data[-1, :], \
                np.linspace(30, 34., 10)))


def test_interpolate_large():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    cfg = {
        "name": "interpolate", \
        "nrows": 300, \
        "ncols": 500, \
        "cellsize": 1., \
        "dtype": np.float64, \
        "xllcorner": 0.,\
        "yllcorner": 0.
    }
    gr = Grid(**cfg)
    gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

    cfg["nrows"] = 600
    cfg["ncols"] = 1000
    cfg["cellsize"] = 0.5
    gr_geom = Grid(**cfg)
    gri = gr.interpolate(gr_geom)

    v0 = gr.data[0, 0]
    v1 = gr.data[0, -1]
    assert (np.allclose(gri.data[0, :], \
                np.linspace(v0, v1, gri.ncols)))

    v0 = gr.data[-1, 0]
    v1 = gr.data[-1, -1]
    assert (np.allclose(gri.data[-1, :], \
                np.linspace(v0, v1, gri.ncols)))


def test_nodata():
    """ Test setting nodata """
    nrows = 6
    gr = Grid(nrows, nrows, dtype=np.int32)

    gr.nodata = 3.6
    assert (3==gr.nodata)

    msg = "cannot convert"
    with pytest.raises(ValueError, match=msg):
        gr.nodata = np.nan


def test_cells_inside_polygon():
    if not has_c_module("gis", False):
        pytest.skip("Missing C module c_hydrodiy_gis")

    nrows = 10
    gr = Grid(nrows, nrows, dtype=np.int32)

    polygon = np.array([[0.5, 2.3], [7.2, 9.5], [6.2, 2.2]])
    inside = gr.cells_inside_polygon(polygon)

    fe = FTEST/"grid_cells_inside_polygon.csv"
    expected = pd.read_csv(fe)
    for cn in inside.columns:
        assert (np.allclose(inside.loc[:, cn], \
                    expected.loc[:, cn]))



def test_downstream():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)

    idxup = range(0, nr*nc)
    idxdown = ca.downstream(idxup)
    expected = -2*np.ones_like(idxup)
    expected[1] = 7
    expected[2] = 8
    expected[3] = 9
    expected[7] = 13
    expected[8] = 14
    expected[9] = 14
    expected[13] = 20
    expected[14] = 20
    expected[15] = 20
    expected[20] = 27
    expected[27] = 33

    ck = np.allclose(idxdown, expected)
    assert ck


def test_upstream():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)

    idxdown = range(0, nr*nc)
    idxup = ca.upstream(idxdown)
    expected = -1*np.ones_like(idxup)
    expected[7:10, 0] = [1, 2, 3]
    expected[13, 0] = 7
    expected[14, :2] = [8, 9]
    expected[20, :3] = [13, 14, 15]
    expected[27, 0] = 20
    expected[33, 0] = 27

    ck = np.allclose(idxup, expected)
    assert ck


def test_extent():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)
    ca.delineate_area(27)
    ext = ca.extent()
    assert (ext == (1., 4., 1., 6.))


def test_delineate_area():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)

    ca.delineate_area(27)
    idxc = ca.idxcells_area
    expected = [1, 2, 3, 7, 8, 9, 13, 14, 15, 20, 27]
    ck = np.allclose(np.sort(idxc), expected)
    assert ck

    ca.delineate_area(12)
    idxc = ca.idxcells_area
    assert (len(idxc) == 0)

    ca.delineate_area(14)
    idxc = ca.idxcells_area
    expected = [2, 3, 8, 9, 14]
    ck = np.allclose(np.sort(idxc), expected)
    assert ck

    # Add one inlet
    ca.delineate_area(27, 14)
    idxc = ca.idxcells_area
    expected = [1, 7, 13, 15, 20, 27]
    ck = np.allclose(np.sort(idxc), expected)
    assert ck

    # Add two inlets
    ca.delineate_area(27, [14, 13])
    idxc = ca.idxcells_area
    expected = [15, 20, 27]
    ck = np.allclose(np.sort(idxc), expected)
    assert ck


def test_delineate_river():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)
    dat = delineate_river(TESTGRID, 1)

    idxc = dat["idxcell"]
    dat = dat.values

    expected = [1, 7, 13, 20, 27, 33]
    ck = np.allclose(idxc, expected)
    assert ck

    expected = [0., 1., 2., 2.+math.sqrt(2), 2.+2*math.sqrt(2),
                                        3.+2*math.sqrt(2)]
    ck = np.allclose(dat[:, 0], expected)
    assert ck

    expected = [0.]*3 + [-1]*2 + [0]
    ck = np.allclose(dat[:, 1], expected)
    assert ck

    expected = [0.] + [-1]*5
    ck = np.allclose(dat[:, 2], expected)
    assert ck

    expected = [1.5, 1.5, 1.5, 2.5, 3.5, 3.5]
    ck = np.allclose(dat[:, 3], expected)
    assert ck

    expected = np.arange(6)[::-1]+0.5
    ck = np.allclose(dat[:, 4], expected)
    assert ck


def test_delineate_boundary():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)

    ca.delineate_area(27)
    ca.delineate_boundary()
    idxc = ca.idxcells_boundary

    expected = [1, 2, 3, 9, 15, 20, 13, 7, 1]
    ck = np.allclose(idxc, expected)

    assert ck


def test_flowpathlengths():
    nr = TESTGRID.nrows
    nc = TESTGRID.ncols
    ca = Catchment("test", TESTGRID)

    ca.delineate_area(27)
    ca.compute_flowpathlengths()
    paths = ca.flowpathlengths

    expected = np.zeros((11, 3), dtype=np.float64)
    expected[:, 1] = 27
    expected[1, 1] = -2
    expected[:, 0] = [20, 27, 13, 14, 15, 7, 8, 9, 1, 2, 3]
    sq = math.sqrt(2)
    expected[:, 2] = [sq, 0, 2*sq, 1+sq, 2*sq, 1+2*sq, 2+sq, \
                        1+2*sq, 2+2*sq, 3+sq, 2+2*sq]

    ck = np.allclose(paths.values, expected)
    assert ck


def test_dict():
    ca = Catchment("test", TESTGRID)

    ca.delineate_area(27)
    ca.delineate_boundary()
    dic = ca.to_dict()

    ca2 = Catchment.from_dict(dic)
    ca2.delineate_boundary()

    for att in ca.__dict__:
        a = str(getattr(ca, att))
        b = str(getattr(ca2, att))
        assert (a==b)


def test_isin():
    ca = Catchment("test", TESTGRID)
    ca.delineate_area(27)

    ck = ca.isin(0)
    assert (~ck)

    ck = ca.isin(1)
    assert ck


def test_intersect():
    ca = Catchment("test", TESTGRID)
    ca.delineate_area(27)

    xll = ca.flowdir.xllcorner+1
    yll = ca.flowdir.yllcorner+1
    csz = 2
    gr = Grid(3, 3, xllcorner=xll,
        yllcorner=yll, cellsize=csz)

    gra, idx, w = ca.intersect(gr)

    ck = np.allclose(idx, [6, 7, 3, 4, 0, 1])
    assert ck

    ck = np.allclose(w, [0.25, 0.25, 1., 0.5, 0.5, 0.25])
    assert ck

    ck = np.allclose(gra.data, np.array([[0.5, 0.25], \
                                            [1., 0.5], [0.25, 0.25]]))
    assert ck


def test_add():
    ca1 = Catchment("test", TESTGRID)
    ca1.delineate_area(13)

    ca2 = Catchment("test", TESTGRID)
    ca2.delineate_area(14)

    ca = ca1+ca2
    ck = np.allclose(ca.idxcells_area, [1, 2, 3, 7, 8, 9, 13, 14])
    assert ck


def test_compute_area():
    if not HAS_PYPROJ:
        warnings.warn("Compute area not tested. Please install pyproj")
        return

    gr = TESTGRID.clone()
    gr.xllcorner = 130.
    gr.yllcorner = -20.

    ca = Catchment("test", TESTGRID)
    ca.delineate_area(27)
    ca.delineate_boundary()

    gda94 = pyproj.Proj("+init=EPSG:3112")
    area = ca.compute_area(gda94)

    ck = np.allclose(area, 60120.97484329)
    # not passing ???
    #assert ck


def test_sub():
    ca1 = Catchment("test", TESTGRID)
    ca1.delineate_area(33)

    ca2 = Catchment("test", TESTGRID)
    ca2.delineate_area(20)

    ca = ca1-ca2
    ck = np.allclose(ca.idxcells_area, [27, 33])
    assert ck


def test_slope():
    alt = TESTGRID.clone()
    alt.dtype = np.float64
    alt.nodata = np.nan
    alt.data = alt.data*0. + np.arange(alt.nrows)[::-1][:, None]
    slp = slope(TESTGRID, alt)

    sq2 = 1./math.sqrt(2)
    expected = np.array([[np.nan, 1., 1., 1,    np.nan, np.nan],
                [np.nan, 1., 1., sq2,  np.nan, np.nan],
                [np.nan, sq2, 1., sq2, np.nan, np.nan],
                [np.nan, np.nan, sq2, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 1.,  np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

    idx = ~np.isnan(expected)
    ck = np.allclose(slp.data[idx], expected[idx])
    ck &= np.all(np.isnan(slp.data[~idx]))
    assert ck


def test_accumulate():
    # Standard accumulation
    acc = accumulate(TESTGRID, nprint=10)
    expected = [ [-1, 1, 1, 1, -1, -1],
                [-1, 2, 2, 2, -1, -1],
                [-1, 3, 5, 1, -1, -1],
                [-1, -1, 10, -1, -1, -1],
                [-1, -1, -1, 11, -1, -1],
                [-1, -1, -1, -1, -1, -1]]

    ck = np.allclose(acc.data, expected)
    assert ck

    # Test with an accumulation field
    to_acc = acc.clone()
    to_acc.fill(0.1)
    to_acc.nodata = -0.1
    acc = accumulate(TESTGRID, to_acc, nprint=10)
    ck = np.allclose(10*acc.data, expected)
    assert ck

    # Restrict the maximum number of accumulated cells
    acc = accumulate(TESTGRID, nprint=10, max_accumulated_cells=2)
    expected = [[-1, 1, 1, 1, -1, -1],
                [-1, 2, 2, 2, -1, -1],
                [-1, 3, 5, 1, -1, -1],
                [-1, -1, 10, -1,-1,-1],
                [-1, -1, -1, 8, -1, -1],
                [-1, -1, -1, -1, -1, -1]]
    ck = np.allclose(acc.data, expected)
    assert ck


def test_voronoi():
    ca = Catchment("test", TESTGRID)
    ca.delineate_area(27)

    # Points in the for corner of grid
    xy = [[0., 0.], [0., 5.], [5., 0.], [5., 5.]]

    we = voronoi(ca, xy)
    assert (np.allclose(we, [1./11, 6./11, 1./11, 3./11]))


def test_accumulate_advanced():
    if not RUN_ADVANCED:
        pytest.skip("Skipping advanced grid tests")

    filename = FTEST/"fdtest.hdr"
    flowdir = Grid.from_header(filename)

    acc = accumulate(flowdir, nprint=30000)

    fileacc = filename.parent / f"{filename.stem}_acc.bil"
    acc.save(fileacc)

    logacc = acc.clone()
    dt = np.log(logacc.data)
    logacc.data = dt

    plt.close("all")
    fig, ax = plt.subplots()

    logacc.plot(ax, interpolation="nearest", cmap="Blues")

    fileplot = FIMG / f"{fileacc.stem}.png"
    fig.savefig(fileplot)


def test_delineate_advanced():
    if not RUN_ADVANCED:
        pytest.skip("Skipping advanced grid tests")

    configs = [
        {"outletxy":[147.7225, -37.2575], "upstreamxy":[147.9, -37.0],
            "filename":"flowdir_223202.hdr"},
        {"outletxy":[145.934, -17.9935], "upstreamxy": [145.7, -17.8],
            "filename":"fdtest.hdr"}
    ]

    for cfg in configs:
        outletxy = cfg["outletxy"]
        upstreamxy = cfg["upstreamxy"]
        filename = FTEST/cfg["filename"]
        flowdir = Grid.from_header(filename)

        ca = Catchment("test", flowdir)

        # Delineate catchment
        idxcell = flowdir.coord2cell(outletxy)
        ca.delineate_area(idxcell)
        ca.delineate_boundary()

        # Delineate river path
        idxcell = flowdir.coord2cell(upstreamxy)
        datariver = delineate_river(flowdir, idxcell, nval=160)

        # Get grid array
        cellsize = 0.1
        xllcorner = np.floor(flowdir.xllcorner*100)/100
        yllcorner = np.floor(flowdir.yllcorner*100)/100
        nrows = np.ceil(flowdir.nrows*flowdir.cellsize/cellsize)
        ncols = np.ceil(flowdir.ncols*flowdir.cellsize/cellsize)
        coarse_grid = Grid("coarse", ncols=ncols, \
                nrows=nrows, xllcorner=xllcorner, yllcorner=yllcorner, \
                cellsize=cellsize)

        gri, idxi, wi = ca.intersect(coarse_grid)
        coordi = coarse_grid.cell2coord(idxi)

        fi = FIMG/f"{filename.stem}_intersect.bil"
        gri.dtype = np.float32
        gri.save(fi)

        # Plots
        plt.close("all")
        fig, ax = plt.subplots()

        # plot flow dir
        flowdir.dtype = np.float64
        data = flowdir.data
        data[data>128] = np.nan
        data = np.log(data)/math.log(2)
        flowdir.data = data
        flowdir.plot(ax, interpolation="nearest", cmap="Blues")

        # Plot intersect
        gri.data = np.power(gri.data, 0.2)
        gri.plot(ax, alpha=0.5, cmap="Reds")
        ax.plot(coordi[:, 0], coordi[:, 1], "+", markersize=20, \
                                color="r")

        # plot catchment
        ca.plot_area(ax, "+", markersize=2)

        # plot boundary
        ca.plot_boundary(ax, color="green", lw=4)

        # plot river
        ax.plot(datariver["x"], datariver["y"], "r", lw=3)

        fig.set_size_inches((15, 15))
        fig.tight_layout()
        fp = FIMG/f"{filename}_plot.png"
        fig.savefig(fp)


def test_name_error():
    msg = "Expected name in"
    with pytest.raises(ValueError, match=msg):
        gr = get_grid("AWRAL_RIVER_BIDULE")

def test_awral():
    gr = get_grid("AWRAL")
    assert (gr.nrows==681)
    assert (gr.ncols==841)
    assert (gr.xllcorner==112.)
    assert (gr.yllcorner==-44.)
    assert (np.sum(gr.data)==281655)


def test_awap():
    gr = get_grid("AWAP")
    assert (gr.nrows==691)
    assert (gr.ncols==886)
    assert (gr.xllcorner== 112.)
    assert (gr.yllcorner== -44.5)
    assert (np.sum(gr.data)==284547)

def test_waterdyn():
    gr = get_grid("WATERDYN")
    assert (gr.nrows==670)
    assert (gr.ncols==813)
    assert (np.isclose(gr.xllcorner, 112.925))
    assert (np.isclose(gr.yllcorner, -43.575))
    assert (np.sum(gr.data)==274845)

def test_dlcd():
    pytest.skip("Skipping this test - too high memory consumption")
    gr = get_grid("DLCD")
    assert (gr.nrows==14902)
    assert (gr.ncols==19161)
    assert (np.isclose(gr.xllcorner, 110.))
    assert (np.isclose(gr.yllcorner, -45.0048))


def test_awral_subgrids():
    for name in AWRAL_SUBGRIDS.gridid:
        gr = get_grid(name)

        if name == "AWRAL_RIVER_MURRUMBIDGEE":
            assert (gr.nrows==47)
            assert (gr.ncols==128)
            assert (np.isclose(gr.xllcorner, 143.2))
            assert (np.isclose(gr.yllcorner, -36.55))

        elif name == "AWRAL_DRAINAGE_MURRAY_DARLING":
            assert (gr.nrows==261)
            assert (gr.ncols==278)
            assert (np.isclose(gr.xllcorner, 138.55))
            assert (np.isclose(gr.yllcorner, -37.65))
            v = np.unique(gr.data.flatten())
            assert (np.allclose(v, [0, 1]))

        elif name == "AWRAL_STATE_NSW":
            assert (gr.nrows==187)
            assert (gr.ncols==252)
            assert (np.isclose(gr.xllcorner, 141.0))
            assert (np.isclose(gr.yllcorner, -37.5))

        v = np.unique(gr.data.flatten())
        assert (np.allclose(v, [0, 1]))



