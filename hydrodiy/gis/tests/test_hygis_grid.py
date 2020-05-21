import os, re, math
import warnings
import unittest
import numpy as np
import pandas as pd
import warnings
import zipfile
from scipy.spatial.distance import pdist, squareform

import zipfile

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from hydrodiy.gis.grid import Grid, Catchment, HAS_C_GIS_MODULE
from hydrodiy.gis.grid import accumulate, voronoi, delineate_river, slope
from hydrodiy.gis.grid import get_mask, AWRAL_SUBGRIDS
from hydrodiy.io import csv

source_file = os.path.abspath(__file__)

RUN_ADVANCED = True


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
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

    def test_print(self):
        gr = Grid(**self.config)
        print(gr)


    def test_name_comment(self):
        gr = Grid(**self.config)
        self.assertTrue(gr.comment == self.config['comment'])
        self.assertTrue(gr.name == self.config['name'])


    def test_shape(self):
        gr = Grid(**self.config)
        self.assertEqual(gr.shape, (7, 5))


    def test_dtype(self):
        gr = Grid(**self.config)
        gr.data = np.random.uniform(0, 1, (gr.nrows, gr.ncols))

        gr.dtype = np.int32
        ck = np.allclose(gr.data, 0.)
        self.assertTrue(ck)


    def test_xvalues_yvalues(self):
        ''' test x and y coords of grid '''
        gr = Grid(**self.config)

        xv, yv = gr.xvalues, gr.yvalues
        self.assertTrue(np.allclose(xv, np.arange(131, 141, 2)))
        self.assertTrue(np.allclose(yv, np.arange(-38, -24, 2)[::-1]))


    def test_clone(self):
        gr = Grid(**self.config)
        data_ini = np.random.uniform(0, 1, (gr.nrows, gr.ncols))
        gr.data = data_ini.copy()

        # Test data is copied accross
        grc = gr.clone()
        ck = np.allclose(grc.data, gr.data)
        self.assertTrue(ck)

        # Check the original grid is not changing
        grc.data[0, :] = 100
        ck = np.allclose(gr.data, data_ini)
        self.assertTrue(ck)

        # Test cloning works when changing type
        grci = gr.clone(np.int32)
        ck = np.allclose(grci.data, 0)
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
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

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
        ''' Test coord2cell and cell2coord '''
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        gr = Grid(**self.config)

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
        self.assertTrue(ck)

        # Get cell coordinate from index and test
        xycoords2 = gr.cell2coord(idxcell)
        ck = np.allclose(xycoords0+csz/2, xycoords2)
        self.assertTrue(ck)

        rowcol = gr.cell2rowcol(idxcell)
        cka = np.allclose(rowcol[:, 0], np.repeat(np.arange(7), 5))
        ckb = np.allclose(rowcol[:, 1], np.concatenate([np.arange(5)]*7))
        self.assertTrue(cka & ckb)


    def test_slice(self):
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        ndim = 11
        gr = Grid('test', ndim)
        vect = np.arange(0, int(ndim/2)+1)+1.
        vect = np.concatenate([vect[::-1], vect[1:]])
        gr.data = np.array([vect+i for i in range(ndim)])

        # Slicing a valley...
        xyslice = np.vstack([np.arange(1, 11)+0.1, [5.2]*10]).T
        zslice = gr.slice(xyslice)

        expect = [ 10.7,   9.7,   8.7,   7.7,   6.7,   6.9, \
                        7.9,   8.9,   9.9,  10.9]
        ck = np.allclose(zslice, expect)
        self.assertTrue(ck)


    def test_from_stream(self):
        ''' Test loading grid from stream '''
        # Header only
        fileheader = os.path.join(self.ftest, 'fdtest.hdr')
        with open(fileheader, 'r') as fh:
            gr = Grid.from_stream(fh)

        def test_grid(gr):
            ck = gr.nrows == 283
            ck &= (gr.ncols == 293)
            ck &= np.isclose(gr.xllcorner, 145.44625)
            ck &= np.isclose(gr.yllcorner, -18.291250)
            ck &= np.isclose(gr.cellsize,  0.0025)
            ck &= (gr.nodata == 32767)
            ck &= (gr.name == 'fdtest')
            ck &= (gr.comment == 'No comment')
            return ck

        self.assertTrue(test_grid(gr))

        # Header and data
        filedata = os.path.join(self.ftest, 'fdtest.bil')

        with open(fileheader, 'r') as fh, open(filedata, 'rb') as fd:
            gr = Grid.from_stream(fh, fd)

        self.assertTrue(test_grid(gr))


    def test_from_zip(self):
        ''' Test loading grid from stream piped from zipfile '''
        filezip = os.path.join(self.ftest, 'flowdir_223202.zip')
        fileheader = 'subdir/flowdir_223202.hdr'
        gr = Grid.from_zip(filezip, fileheader)

        ck = gr.nrows == 241
        ck &= (gr.ncols == 260)
        ck &= np.isclose(gr.xllcorner, 147.45125)
        ck &= np.isclose(gr.yllcorner, -37.45125)
        ck &= np.isclose(gr.cellsize,  0.0025)
        ck &= (gr.nodata == 0)
        ck &= (gr.name == 'no_name')
        ck &= (gr.comment == 'No comment')
        ck &= np.isclose(gr.data.mean(), 30.12049154165337)
        self.assertTrue(ck)


    def test_from_header(self):
        filename = os.path.join(self.ftest, 'header.hdr')
        try:
            gr = Grid.from_header(filename)
        except MemoryError:
            warnings.warn('test_from_header not run due to memory error')
            return

        ck = gr.nrows == 13857
        ck &= (gr.ncols == 16440)
        ck &= (gr.xllcorner == 112.90125)
        ck &= (gr.yllcorner == -43.74375)
        ck &= (gr.cellsize == 0.0025)
        ck &= (gr.nodata == 32767)
        ck &= (gr.name == 'mygrid')
        ck &= (gr.comment == 'testing header')

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
        fp = os.path.join(self.fimg, \
                        re.sub('hdr', 'png', os.path.basename(filename)))
        fig.savefig(fp)


    def test_plot_values(self):
        ''' Test showing grid values '''
        ngrid = 20
        gr = Grid('test', xllcorner=0, yllcorner=0, cellsize=1, \
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

        plt.close('all')
        fig, ax = plt.subplots()
        gr.plot(ax, interpolation='nearest')
        gr.plot_values(ax, fmt='0.1f', fontsize=7, fontweight='bold', \
                                            color='w')

        fig.set_size_inches((10, 10))
        fig.tight_layout()
        fp = os.path.join(self.fimg, 'test_plot_values.png')
        fig.savefig(fp)


    def test_apply(self):
        gr = Grid(**self.config)
        gr = gr.apply(lambda x: np.random.uniform(0, 1, x.shape))

        grlog = gr.apply(np.log)
        ck = np.allclose(grlog.data, np.log(gr.data))

        self.assertTrue(ck)


    def test_clip(self):
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        gr = Grid(**self.config)
        gr.data = np.arange(gr.nrows*gr.ncols).reshape((gr.nrows, gr.ncols))
        xll, yll = 132.1, -35.1
        xur, yur = 137.6, -26.9
        grclip = gr.clip(xll, yll, xur, yur)

        expected = gr.data[0:6, 1:4]
        self.assertTrue(np.allclose(expected, grclip.data))
        self.assertTrue((grclip.xlim[0] < xll) & (grclip.xlim[1] > xur))
        self.assertTrue((grclip.ylim[0] < yll) & (grclip.ylim[1] > yur))


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
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

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
        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

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


    def test_cells_inside_polygon(self):
        ''' Test cells inside polygon algorithm '''
        nrows = 10
        gr = Grid(nrows, nrows, dtype=np.int32)

        polygon = np.array([[0.5, 2.3], [7.2, 9.5], [6.2, 2.2]])
        inside = gr.cells_inside_polygon(polygon)

        fe = os.path.join(self.ftest, 'grid_cells_inside_polygon.csv')
        expected = pd.read_csv(fe)

        self.assertTrue(np.allclose(inside, expected))



class CatchmentTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> CatchmentTestCase')

        if not HAS_C_GIS_MODULE:
            self.skipTest('Missing C module c_hydrodiy_gis')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)
        self.fimg = os.path.join(self.ftest, 'images')
        if not os.path.exists(self.fimg):
            os.mkdir(self.fimg)

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
        self.assertEqual(ext, (1., 4., 1., 6.))


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

        expected = [1.5, 1.5, 1.5, 2.5, 3.5, 3.5]
        ck = np.allclose(dat[:, 3], expected)
        self.assertTrue(ck)

        expected = np.arange(6)[::-1]+0.5
        ck = np.allclose(dat[:, 4], expected)
        self.assertTrue(ck)


    def test_delineate_boundary(self):
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

        ca.delineate_area(27)
        ca.delineate_boundary()
        idxc = ca.idxcells_boundary

        expected = [1, 2, 3, 9, 15, 20, 13, 7, 1]
        ck = np.allclose(idxc, expected)

        self.assertTrue(ck)


    def test_flowpathlengths(self):
        ''' Test flow path delineation '''
        nr = self.gr.nrows
        nc = self.gr.ncols
        ca = Catchment('test', self.gr)

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
        self.assertTrue(ck)


    def test_dict(self):
        ''' test conversion to dict '''
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

        xll = ca.flowdir.xllcorner+1
        yll = ca.flowdir.yllcorner+1
        csz = 2
        gr = Grid(3, 3, xllcorner=xll,
            yllcorner=yll, cellsize=csz)

        gra, idx, w = ca.intersect(gr)

        ck = np.allclose(idx, [6, 7, 3, 4, 0, 1])
        self.assertTrue(ck)

        ck = np.allclose(w, [0.25, 0.25, 1., 0.5, 0.5, 0.25])
        self.assertTrue(ck)

        ck = np.allclose(gra.data, np.array([[0.5, 0.25], \
                                                [1., 0.5], [0.25, 0.25]]))
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

        ck = np.allclose(area, 60120.97484329)
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
        if not RUN_ADVANCED:
            self.skipTest('Skipping advanced grid tests')

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

        fileplot = os.path.join(self.fimg, \
                        re.sub('\\.bil', '.png', os.path.basename(fileacc)))
        fig.savefig(fileplot)


    def test_delineate_advanced(self):
        if not RUN_ADVANCED:
            self.skipTest('Skipping advanced grid tests')

        configs = [
            {'outletxy':[147.7225, -37.2575], 'upstreamxy':[147.9, -37.0],
                'filename':'flowdir_223202.hdr'},
            {'outletxy':[145.934, -17.9935], 'upstreamxy': [145.7, -17.8],
                'filename':'fdtest.hdr'}
        ]

        for cfg in configs:
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

            # Get grid array
            cellsize = 0.1
            xllcorner = np.floor(flowdir.xllcorner*100)/100
            yllcorner = np.floor(flowdir.yllcorner*100)/100
            nrows = np.ceil(flowdir.nrows*flowdir.cellsize/cellsize)
            ncols = np.ceil(flowdir.ncols*flowdir.cellsize/cellsize)
            coarse_grid = Grid('coarse', ncols=ncols, \
                    nrows=nrows, xllcorner=xllcorner, yllcorner=yllcorner, \
                    cellsize=cellsize)

            gri, idxi, wi = ca.intersect(coarse_grid)
            coordi = coarse_grid.cell2coord(idxi)

            fi = os.path.join(self.fimg, \
                        re.sub('\\.hdr', '_intersect.bil', \
                            os.path.basename(filename)))
            gri.dtype = np.float32
            gri.save(fi)

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

            # Plot intersect
            gri.data = np.power(gri.data, 0.2)
            gri.plot(ax, alpha=0.5, cmap='Reds')
            ax.plot(coordi[:, 0], coordi[:, 1], '+', markersize=20, \
                                    color='r')

            # plot catchment
            ca.plot_area(ax, '+', markersize=2)

            # plot boundary
            ca.plot_boundary(ax, color='green', lw=4)

            # plot river
            ax.plot(datariver['x'], datariver['y'], 'r', lw=3)

            fig.set_size_inches((15, 15))
            fig.tight_layout()
            fp = os.path.join(self.fimg, \
                        re.sub('\\.hdr', '_plot.png', \
                            os.path.basename(filename)))
            fig.savefig(fp)



class RefGridsTestCase(unittest.TestCase):

    def setUp(self):
        print('\t=> RefGridsTestCase')

        source_file = os.path.abspath(__file__)
        self.ftest = os.path.dirname(source_file)


    def test_name_error(self):
        ''' Test mask error '''
        try:
            gr = get_mask('AWRAL_RIVER_BIDULE')
        except ValueError as err:
            self.assertTrue(str(err).startswith('Expected name in'))

    def test_awral(self):
        ''' Test awral mask '''
        gr = get_mask('AWRAL')
        self.assertEqual(gr.nrows, 681)
        self.assertEqual(gr.ncols, 841)
        self.assertEqual(gr.xllcorner, 112.)
        self.assertEqual(gr.yllcorner, -44.)
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
        self.assertTrue(np.isclose(gr.xllcorner, 112.925))
        self.assertTrue(np.isclose(gr.yllcorner, -43.575))
        self.assertEqual(np.sum(gr.data), 274845)

    def test_dlcd(self):
        ''' Test DLCD mask '''
        self.skipTest('Skipping this test - too high memory consumption')

        gr = get_mask('DLCD')
        self.assertEqual(gr.nrows, 14902)
        self.assertEqual(gr.ncols, 19161)
        self.assertTrue(np.isclose(gr.xllcorner, 110.))
        self.assertTrue(np.isclose(gr.yllcorner, -45.0048))


    def test_awral_subgrids(self):
        ''' Test awral subgrids mask '''
        for name in AWRAL_SUBGRIDS.gridid:
            gr = get_mask(name)

            if name == 'AWRAL_RIVER_MURRUMBIDGEE':
                self.assertEqual(gr.nrows, 47)
                self.assertEqual(gr.ncols, 128)
                self.assertTrue(np.isclose(gr.xllcorner, 143.2))
                self.assertTrue(np.isclose(gr.yllcorner, -36.55))
                v = np.unique(gr.data.flatten())
                self.assertTrue(np.allclose(v, [0, 1]))

            elif name == 'AWRAL_DRAINAGE_MURRAY_DARLING':
                self.assertEqual(gr.nrows, 261)
                self.assertEqual(gr.ncols, 278)
                self.assertTrue(np.isclose(gr.xllcorner, 138.55))
                self.assertTrue(np.isclose(gr.yllcorner, -37.65))
                v = np.unique(gr.data.flatten())
                self.assertTrue(np.allclose(v, [0, 1]))


if __name__ == "__main__":
    unittest.main()

