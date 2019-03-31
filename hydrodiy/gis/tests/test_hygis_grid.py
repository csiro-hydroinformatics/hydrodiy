import os, re, math
import warnings
import unittest
import numpy as np
import pandas as pd
import warnings

import zipfile

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from hydrodiy.gis.grid import Grid, Catchment
from hydrodiy.gis.grid import accumulate, voronoi, delineate_river, slope
from hydrodiy.gis.grid import get_mask
from hydrodiy.io import csv

source_file = os.path.abspath(__file__)

run_advanced = True


class GridTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> GridTestCase')

        self.config = {'name':'test', \
                'nrows':7, 'ncols':5, 'cellsize':2., \
                'dtype':np.float64, \
                'xllcorner':130., \
                'yllcorner':-39., \
                'comment': 'this is a test grid'}

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_print(self):
        gr = Grid(**self.config)
        print(gr)


    def test_name_comment(self):
        gr = Grid(**self.config)
        self.assertTrue(gr.comment == self.config['comment'])
        self.assertTrue(gr.name == self.config['name'])


    def test_dtype(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

        gr.dtype = np.int32
        ck = np.allclose(gr.data, 0.)
        self.assertTrue(ck)


    def test_clone(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

        grc = gr.clone(np.int32)
        ck = np.allclose(grc.data, 0.)
        ck = ck & (gr.data.shape == grc.data.shape)
        self.assertTrue(ck)


    def test_same_geometry(self):
        gr = Grid(**self.config)

        grc = gr.clone(np.int32)
        self.assertTrue(gr.same_geometry(grc))

        grc = Grid('test', ncols=10)
        self.assertTrue(~gr.same_geometry(grc))


    def test_getitem(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

        idx = [0, 2, 5]
        val = gr[idx]
        self.assertTrue(np.allclose(val, gr.data.flat[idx]))


    def test_setitem(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(10, 11, (gr.nrows, gr.ncols))

        idx = [0, 2, 5]
        val = np.arange(len(idx))
        gr[idx] = val
        self.assertTrue(np.allclose(val, gr.data.flat[idx]))


    def test_data(self):
        gr = Grid(**self.config)

        dt = np.random.uniform(0, 1, (gr.nrows, gr.ncols))
        gr.data = dt
        ck = np.allclose(gr.data, dt)
        self.assertTrue(ck)

        dt = np.random.uniform(0, 1, (gr.nrows+1, gr.ncols+1))
        try:
            gr.data = dt
        except ValueError as err:
            self.assertTrue(str(err).startswith('Wrong number'))
        else:
            raise Exception('Problem with handling data error')

    def test_fill(self):
        gr = Grid(**self.config)
        gr.fill(-2.)

        ck = np.allclose(gr.data, -2.)
        self.assertTrue(ck)


    def test_save(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(0, 1, \
                        (gr.nrows, gr.ncols))

        # Write data
        fg = os.path.join(self.ftest, 'grid_test_save.bil')
        gr.save(fg)

        # Load it back
        gr2 = Grid.from_header(fg)

        ck = np.allclose(gr.data, gr2.data)
        self.assertTrue(ck)

        self.assertTrue(self.config['comment'] == gr2.comment)


    def test_dict(self):
        gr = Grid(**self.config)
        js = gr.to_dict()
        self.assertEqual(js['dtype'], '<f8')

        gr2 = Grid.from_dict(js)
        for att in gr.__dict__:
            a = str(getattr(gr, att))
            b = str(getattr(gr2, att))
            self.assertEqual(a, b)


    def test_neighbours(self):
        gr = Grid(**self.config)

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
            self.assertTrue(ck)

        # Upper left corner
        nb = gr.neighbours(0)
        ck = np.allclose(nb, [-1, -1, -1, -1, -1, 1, -1, nc, nc+1])
        self.assertTrue(ck)

        # Lower right corner
        idxcell = nr*nc-1
        nb = gr.neighbours(nr*nc-1)
        ck = np.allclose(nb, [idxcell-nc-1, idxcell-nc, -1,
            idxcell-1, -1, -1, -1, -1, -1])
        self.assertTrue(ck)

        # Error
        try:
            nb = gr.neighbours(nr*nc)
        except ValueError as err:
            self.assertTrue(str(err).startswith('c_hydrodiy_gis.neighbours'))
        else:
            raise Exception('Problem with handling of error')


    def test_coord2cell(self):
        gr = Grid(**self.config)

        csz = gr.cellsize
        xll = gr.xllcorner
        yll = gr.yllcorner

        xx = xll+np.arange(0, gr.ncols)*csz
        yy = yll+np.arange(gr.nrows-1, -1, -1)*csz
        xxg, yyg = np.meshgrid(xx, yy)
        xycoords0 = np.concatenate([xxg.flat[:][:, None],
                    yyg.flat[:][:, None]], axis=1)
        xycoords1 = xycoords0 + np.random.uniform(-csz/2, csz/2,
                        xycoords0.shape)
        idxcell = gr.coord2cell(xycoords1)
        ck = np.allclose(idxcell, np.arange(gr.ncols*gr.nrows))
        self.assertTrue(ck)

        xycoords2 = gr.cell2coord(idxcell)
        ck = np.allclose(xycoords0, xycoords2)
        self.assertTrue(ck)


    def test_slice(self):
        ndim = 11
        gr = Grid('test', ndim)
        vect = np.arange(0, int(ndim/2)+1)+1.
        vect = np.concatenate([vect[::-1], vect[1:]])
        gr.data = np.repeat(vect.reshape((1, ndim)), ndim, 0)

        # Slicing a valley...
        xyslice = np.vstack([np.arange(0, 11)+0.1, [5.1]*11]).T
        zslice = gr.slice(xyslice)

        expect = vect
        expect[:ndim//2] -= 0.1
        expect[ndim//2:-1] += 0.1
        ck = np.allclose(zslice, expect)
        self.assertTrue(ck)


    def test_from_header(self):
        filename = os.path.join(self.ftest, 'header.hdr')
        try:
            gr = Grid.from_header(filename)
        except MemoryError:
            warnings.warn('test_from_header not run due to memory error')
            return

        ck = gr.nrows == 13857
        ck = ck & (gr.ncols == 16440)
        ck = ck & (gr.xllcorner == 112.90125)
        ck = ck & (gr.yllcorner == -43.74375)
        ck = ck & (gr.cellsize == 0.0025)
        ck = ck & (gr.nodata == 32767)
        ck = ck & (gr.name == 'mygrid')
        ck = ck & (gr.comment == 'testing header')

        self.assertTrue(ck)


    def test_plot(self):
        filename = os.path.join(self.ftest, 'demtest.hdr')
        gr = Grid.from_header(filename)
        gr.dtype = np.float64
        def fun(x):
            idx = x>=55500
            x[idx] = np.nan
            return np.log(x+1)
        gr = gr.apply(fun)

        plt.close('all')
        fig, ax = plt.subplots()
        gr.plot(ax, interpolation='nearest')
        fp = re.sub('hdr', 'png', filename)
        fig.savefig(fp)


    def test_apply(self):
        gr = Grid(**self.config)
        gr = gr.apply(lambda x: np.random.uniform(0, 1, x.shape))

        grlog = gr.apply(np.log)
        ck = np.allclose(grlog.data, np.log(gr.data))

        self.assertTrue(ck)


    def test_clip(self):
        gr = Grid(**self.config)
        gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))
        grclip = gr.clip(132.1, -35.1, 137.6, -26.9)

        expected = gr.data[0:5, 1:5]
        ck = np.allclose(expected, grclip.data)
        self.assertTrue(ck)


    def test_minmaxdata(self):
        cfg = self.config.copy()
        cfg['dtype'] = np.int32
        gr = Grid(**cfg)
        gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

        # Check the minimum is converted to proper dtype
        gr.mindata = 20.4
        self.assertTrue(gr.mindata == 20)
        self.assertTrue(gr.data.min() == 20)

        gr.maxdata = 30.6
        self.assertTrue(gr.maxdata == 30)
        self.assertTrue(gr.data.max() == 30)

        try:
            gr.mindata = 35
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected mindata<maxdata'))
        else:
            raise ValueError('Problem with error handling')


    def test_interpolate_small(self):
        ''' Small grid interpolation '''
        gr = Grid(**self.config)
        gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

        cfg = {
            'name': 'interpolate', \
            'nrows': 14, \
            'ncols': 10, \
            'cellsize': 1.
        }
        for key in ['dtype', 'xllcorner', 'yllcorner']:
            cfg[key] = self.config[key]

        gr_geom = Grid(**cfg)
        gri = gr.interpolate(gr_geom)
        self.assertTrue(np.allclose(gri.data[0, :], \
                    np.linspace(0, 4., 10)))

        self.assertTrue(np.allclose(gri.data[-1, :], \
                    np.linspace(30, 34., 10)))


    def test_interpolate_large(self):
        ''' Large grid interpolation '''

        cfg = {
            'name': 'interpolate', \
            'nrows': 300, \
            'ncols': 500, \
            'cellsize': 1., \
            'dtype': np.float64, \
            'xllcorner': 0.,\
            'yllcorner': 0.
        }
        gr = Grid(**cfg)
        gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))

        cfg['nrows'] = 600
        cfg['ncols'] = 1000
        cfg['cellsize'] = 0.5
        gr_geom = Grid(**cfg)
        gri = gr.interpolate(gr_geom)

        v0 = gr.data[0, 0]
        v1 = gr.data[0, -1]
        self.assertTrue(np.allclose(gri.data[0, :], \
                    np.linspace(v0, v1, gri.ncols)))

        v0 = gr.data[-1, 0]
        v1 = gr.data[-1, -1]
        self.assertTrue(np.allclose(gri.data[-1, :], \
                    np.linspace(v0, v1, gri.ncols)))


    def test_nodata(self):
        ''' Test setting nodata '''
        nrows = 6
        gr = Grid(nrows, nrows, dtype=np.int32)

        gr.nodata = 3.6
        self.assertEqual(3, gr.nodata)

        try:
            gr.nodata = np.nan
        except Exception as err:
            self.assertTrue(str(err).startswith('cannot convert'))
        else:
            raise ValueError('Problem with error generation')



class CatchmentTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CatchmentTestCase')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)

        nrows = 6
        gr = Grid(nrows, nrows, dtype=np.int32, nodata=-1)
        gr.data = [ [0, 4, 4, 4, 0, 0],
                    [0, 4, 4, 8, 0, 0],
                    [0, 2, 4, 8, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
        self.gr = gr

        self.grn = np.arange(nrows*nrows).reshape((nrows, nrows))


    def test_downstream(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

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
        self.assertTrue(ck)


    def test_upstream(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

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
        self.assertTrue(ck)


    def test_extent(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)
        ca.delineate_area(27)
        ext = ca.extent()
        self.assertEqual(ext, (1., 3., 1., 5.))


    def test_delineate_area(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

        ca.delineate_area(27)
        idxc = ca.idxcells_area
        expected = [1, 2, 3, 7, 8, 9, 13, 14, 15, 20, 27]
        ck = np.allclose(np.sort(idxc), expected)
        self.assertTrue(ck)

        ca.delineate_area(12)
        idxc = ca.idxcells_area
        self.assertTrue(len(idxc) == 0)

        ca.delineate_area(14)
        idxc = ca.idxcells_area
        expected = [2, 3, 8, 9, 14]
        ck = np.allclose(np.sort(idxc), expected)
        self.assertTrue(ck)

        # Add one inlet
        ca.delineate_area(27, 14)
        idxc = ca.idxcells_area
        expected = [1, 7, 13, 15, 20, 27]
        ck = np.allclose(np.sort(idxc), expected)
        self.assertTrue(ck)

        # Add two inlets
        ca.delineate_area(27, [14, 13])
        idxc = ca.idxcells_area
        expected = [15, 20, 27]
        ck = np.allclose(np.sort(idxc), expected)
        self.assertTrue(ck)


    def test_delineate_river(self):
        nr = self.gr.nrows
        nc = self.gr.ncols

        dat = delineate_river(self.gr, 1)
        idxc = dat['idxcell']
        dat = dat.values

        expected = [1, 7, 13, 20, 27, 33]
        ck = np.allclose(idxc, expected)
        self.assertTrue(ck)

        expected = [0., 1., 2., 2.+math.sqrt(2), 2.+2*math.sqrt(2),
                                            3.+2*math.sqrt(2)]
        ck = np.allclose(dat[:, 0], expected)
        self.assertTrue(ck)

        expected = [0.]*3 + [-1]*2 + [0]
        ck = np.allclose(dat[:, 1], expected)
        self.assertTrue(ck)

        expected = [0.] + [-1]*5
        ck = np.allclose(dat[:, 2], expected)
        self.assertTrue(ck)

        expected = [1., 1., 1., 2., 3., 3.]
        ck = np.allclose(dat[:, 3], expected)
        self.assertTrue(ck)

        expected = range(6)[::-1]
        ck = np.allclose(dat[:, 4], expected)
        self.assertTrue(ck)


    def test_delineate_boundary(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

        ca.delineate_area(27)
        ca.delineate_boundary()
        idxc = ca.idxcells_boundary

        expected = [1, 7, 13, 20, 15, 9, 3, 1]
        ck = np.allclose(idxc, expected)
        self.assertTrue(ck)


    def test_dic(self):
        ca = Catchment('test', self.gr)

        ca.delineate_area(27)
        ca.delineate_boundary()
        dic = ca.to_dict()

        ca2 = Catchment.from_dict(dic)
        ca2.delineate_boundary()

        for att in ca.__dict__:
            a = str(getattr(ca, att))
            b = str(getattr(ca2, att))
            self.assertEqual(a, b)


    def test_isin(self):
        ca = Catchment('test', self.gr)
        ca.delineate_area(27)

        ck = ca.isin(0)
        self.assertTrue(~ck)

        ck = ca.isin(1)
        self.assertTrue(ck)


    def test_intersect(self):
        ca = Catchment('test', self.gr)
        ca.delineate_area(27)

        xll = self.gr.xllcorner+1
        yll = self.gr.yllcorner+1
        csz = 3
        gr = Grid(2, 2, xllcorner=xll,
            yllcorner=yll, cellsize=csz)

        idx, w = ca.intersect(gr)
        ck = np.allclose(idx, [2, 3, 0, 1])
        ck = ck & np.allclose(w, [1./9, 1./9, 2./3, 1./3])
        self.assertTrue(ck)


    def test_add(self):
        ca1 = Catchment('test', self.gr)
        ca1.delineate_area(13)

        ca2 = Catchment('test', self.gr)
        ca2.delineate_area(14)

        ca = ca1+ca2
        ck = np.allclose(ca.idxcells_area, [1, 2, 3, 7, 8, 9, 13, 14])
        self.assertTrue(ck)


    def test_compute_area(self):
        ''' Test computation of catchment area '''
        if not HAS_PYPROJ:
            warnings.warn('Compute area not tested. Please install pyproj')
            return

        gr = self.gr.clone()
        gr.xllcorner = 130.
        gr.yllcorner = -20.

        ca = Catchment('test', gr)
        ca.delineate_area(27)
        ca.delineate_boundary()

        gda94 = pyproj.Proj('+init=EPSG:3112')
        area = ca.compute_area(gda94)

        ck = np.allclose(area, 59733.200152475583)
        self.assertTrue(ck)


    def test_sub(self):
        ca1 = Catchment('test', self.gr)
        ca1.delineate_area(33)

        ca2 = Catchment('test', self.gr)
        ca2.delineate_area(20)

        ca = ca1-ca2
        ck = np.allclose(ca.idxcells_area, [27, 33])
        self.assertTrue(ck)


    def test_slope(self):
        ''' Test computation of slope '''
        alt = self.gr.clone()
        alt.dtype = np.float64
        alt.nodata = np.nan
        alt.data = alt.data*0. + np.arange(alt.nrows)[::-1][:, None]
        slp = slope(self.gr, alt)

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
        self.assertTrue(ck)


    def test_accumulate(self):
        ''' Test accumulate '''
        # Standard accumulation
        acc = accumulate(self.gr, nprint=10)
        expected = [ [-1, 1, 1, 1, -1, -1],
                    [-1, 2, 2, 2, -1, -1],
                    [-1, 3, 5, 1, -1, -1],
                    [-1, -1, 10, -1, -1, -1],
                    [-1, -1, -1, 11, -1, -1],
                    [-1, -1, -1, -1, -1, -1]]

        ck = np.allclose(acc.data, expected)
        self.assertTrue(ck)

        # Test with an accumulation field
        to_acc = acc.clone()
        to_acc.fill(0.1)
        to_acc.nodata = -0.1
        acc = accumulate(self.gr, to_acc, nprint=10)
        ck = np.allclose(10*acc.data, expected)
        self.assertTrue(ck)

        # Restrict the maximum number of accumulated cells
        acc = accumulate(self.gr, nprint=10, max_accumulated_cells=2)
        expected = [[-1, 1, 1, 1, -1, -1],
                    [-1, 2, 2, 2, -1, -1],
                    [-1, 3, 5, 1, -1, -1],
                    [-1, -1, 10, -1,-1,-1],
                    [-1, -1, -1, 8, -1, -1],
                    [-1, -1, -1, -1, -1, -1]]
        ck = np.allclose(acc.data, expected)
        self.assertTrue(ck)


    def test_voronoi(self):
        ca = Catchment('test', self.gr)
        ca.delineate_area(27)

        # Points in the for corner of grid
        xy = [[0., 0.], [0., 5.], [5., 0.], [5., 5.]]

        we = voronoi(ca, xy)

        self.assertTrue(np.allclose(we, [1./11, 6./11, 1./11, 3./11]))


    def test_accumulate_advanced(self):
        if not run_advanced:
            return

        filename = os.path.join(self.ftest, 'fdtest.hdr')
        flowdir = Grid.from_header(filename)

        acc = accumulate(flowdir, nprint=30000)

        fileacc = re.sub('\\.hdr', '_acc.bil', filename)
        acc.save(fileacc)

        logacc = acc.clone()
        dt = np.log(logacc.data)
        logacc.data = dt

        plt.close('all')
        fig, ax = plt.subplots()

        logacc.plot(ax, interpolation='nearest', cmap='Blues')

        fileplot = re.sub('\\.bil', '.png', fileacc)
        fig.savefig(fileplot)


    def test_delineate_advanced(self):
        if not run_advanced:
            return

        config = [
            {'outletxy':[147.72, -37.26], 'upstreamxy':[147.9, -37.0],
                'filename':'flowdir_223202.hdr'},
            {'outletxy':[145.93375, -17.99375], 'upstreamxy': [145.7, -17.8],
                'filename':'fdtest.hdr'}
        ]

        for cfg in config:

            outletxy = cfg['outletxy']
            upstreamxy = cfg['upstreamxy']
            filename = os.path.join(self.ftest, cfg['filename'])

            flowdir = Grid.from_header(filename)

            ca = Catchment('test', flowdir)

            # Delineate catchment
            idxcell = flowdir.coord2cell(outletxy)
            ca.delineate_area(idxcell)
            ca.delineate_boundary()

            # Delineate river path
            idxcell = flowdir.coord2cell(upstreamxy)
            datariver = delineate_river(flowdir, idxcell, nval=160)

            # Plots
            plt.close('all')
            fig, ax = plt.subplots()

            # plot flow dir
            flowdir.dtype = np.float64
            data = flowdir.data
            data[data>128] = np.nan
            data = np.log(data)/math.log(2)
            flowdir.data = data
            flowdir.plot(ax, interpolation='nearest', cmap='Blues')

            # plot catchment
            ca.plot_area(ax, '+', markersize=2)

            # plot boundary
            ca.plot_boundary(ax, color='green', lw=4)

            # plot river
            ax.plot(datariver['x'], datariver['y'], 'r', lw=3)

            fig.set_size_inches((15, 15))
            fig.tight_layout()
            fp = re.sub('\\.hdr', '_plot.png', filename)
            fig.savefig(fp)


class RefGridsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> RefGridsTestCase')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_awral(self):
        ''' Test awral mask '''
        gr = get_mask('AWRAL')
        self.assertEqual(gr.nrows, 681)
        self.assertEqual(gr.ncols, 841)
        self.assertEqual(gr.xllcorner, 111.975)
        self.assertEqual(gr.yllcorner, -44.025)
        self.assertEqual(np.sum(gr.data), 281655)

    def test_awap(self):
        ''' Test awap mask '''
        gr = get_mask('AWAP')
        self.assertEqual(gr.nrows, 691)
        self.assertEqual(gr.ncols, 886)
        self.assertEqual(gr.xllcorner, 112.)
        self.assertEqual(gr.yllcorner, -44.5)
        self.assertEqual(np.sum(gr.data), 284547)

    def test_waterdyn(self):
        ''' Test waterdyn mask '''
        gr = get_mask('WATERDYN')
        self.assertEqual(gr.nrows, 670)
        self.assertEqual(gr.ncols, 813)
        self.assertEqual(gr.xllcorner, 112.925)
        self.assertEqual(gr.yllcorner, -43.575)
        self.assertEqual(np.sum(gr.data), 274845)


if __name__ == "__main__":
    unittest.main()

