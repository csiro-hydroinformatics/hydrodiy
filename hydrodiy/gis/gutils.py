import numpy as np
import math
import matplotlib.image as mpimg

import requests
import time

from hydrodiy.stat import sutils

# Try to import C code
HAS_C_GIS_MODULE = True
try:
    import c_hydrodiy_gis
except ImportError:
    HAS_C_GIS_MODULE = False


def points_inside_polygon(points, polygon, atol=1e-8):
    '''

    Determines if a set of points are inside a given polygon or not
    see [web ref from where I got the code]
    Note that points on the polygon border are not inside!

    Parameters
    -----------
    points : numpy.array polygon
        Coordinates of points given as 2d numpy array [x,y]
    polygon : numpy.array polygon
        Coordinates of polygon vertices given as 2d numpy array [x,y]
    atol : float
        Tolerance factor for float number identity testing

    Returns
    -----------
    cells_inside : numpy.array
        List of grid cells in this polygon

    :param numpy.array points : A list of points given as a 2d numpy array
    :param numpy.array polygon : A polygon defined by a 2d numpy array [x,y]
    :param float atol : absolute tolerance for float comparison

    '''
    if not HAS_C_GIS_MODULE:
        raise ValueError('C module c_hydrodiy_gis is not available, '+\
            'please run python setup.py build')

    # Prepare inputs
    atol = np.float64(atol)
    points = points.astype(np.float64)
    polygon = polygon.astype(np.float64)
    inside = np.zeros(len(points), dtype=np.int32)

    # run C code
    ierr = c_hydrodiy_gis.points_inside_polygon(atol, points, \
                    polygon, inside)

    if ierr>0:
        raise ValueError('c_hydrodiy_gis.points_inside_polygon '+\
                            'returns '+str(ierr))

    inside = inside.astype(bool)

    #nvert = poly.shape[0]
    #xymin = poly.min(axis=0)
    #xymax = poly.max(axis=0)

    #npt = points.shape[0]
    #inside = np.repeat(False, npt)

    #for idx in range(npt):
    #    x, y = points[idx,:]
    #    x = float(x)
    #    y = float(y)

    #    # Simple check
    #    if x<xymin[0] or y<xymin[1] or x>xymax[0] or y>xymax[1]:
    #        continue

    #    # Advanced algorithm
    #    p1x,p1y = poly[0, :]
    #    p1x = float(p1x)
    #    p1y = float(p1y)

    #    for i in range(nvert+1):
    #        p2x,p2y = poly[i % nvert, :]
    #        p2x = float(p2x)
    #        p2y = float(p2y)

    #        if y > min(p1y,p2y):
    #            if y <= max(p1y,p2y):
    #                if x <= max(p1x,p2x):
    #                    iseq = np.isclose(p1y, p2y, atol=atol, rtol=rtol)
    #                    if not iseq:
    #                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x

    #                    iseq = np.isclose(p1x, p2x, atol=atol, rtol=rtol)
    #                    if iseq or x <= xinters:
    #                        inside[idx] = not inside[idx]

    #        p1x,p1y = p2x,p2y

    return inside

def xy2kml(x, y, fkml, z=None, siteid=None, label=None,
            icon='placemark_circle', scale=1.2):
    ''' Convert a series of x/y points to KML format

    Parameters
    -----------
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y coordinates
    fkml : str
        File path to kml file
    z : numpy.ndarray
        Z coordinates
    siteid : numpy.ndarray
        Id of sites
    label : numpy.ndarray
        Label displayed for each site

    Example
    -----------
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.linspace(0, 1, 10)
    >>> fkml = 'kml_file.kml'
    >>> gutils.xy2kml(x, y, fkml)
    '''

    nval = len(x)
    for v in [y, siteid, label]:
        if not v is None:
            if not len(v) == nval:
                raise ValueError('Expected input size equal to {0}, got {1}'.format(\
                nval, len(v)))

    with open(fkml, 'w') as f:
        # Preamble
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2" '+\
            'xmlns:gx="http://www.google.com/kml/ext/2.2" '+\
            'xmlns:kml="http://www.opengis.net/kml/2.2" '+\
            'xmlns:atom="http://www.w3.org/2005/Atom">\n\n')
        f.write('<Document>\n\n')

        # Icon
        f.write('<Style id="s_icon">\n')
        f.write('\t<IconStyle>\n')
        f.write('\t\t<scale>{0:0.1f}</scale>\n'.format(scale))
        f.write('\t\t<Icon>\n')
        f.write('\t\t\t<href>http://maps.google.com/mapfiles/kml/shapes/{0}.png</href>\n'.format(icon))
        f.write('\t\t</Icon>\n')
        f.write('\t</IconStyle>\n')
        f.write('\t<ListStyle></ListStyle>\n')
        f.write('</Style>\n')
        f.write('<StyleMap id="m_icon">\n')
        f.write('\t<Pair>\n')
        f.write('\t\t<key>normal</key>\n')
        f.write('\t\t<styleUrl>#s_icon</styleUrl>\n')
        f.write('\t</Pair>\n')
        f.write('\t<Pair>\n')
        f.write('\t\t<key>highlight</key>\n')
        f.write('\t\t<styleUrl>#s_icon</styleUrl>\n')
        f.write('\t</Pair>\n')
        f.write('</StyleMap>\n\n')

        # Data
        for i in range(nval):
            f.write('<Placemark>\n')
            f.write('\t<styleUrl>#m_icon</styleUrl>\n')

            if not siteid is None:
                f.write('\t<name>{0}</name>\n'.format(siteid[i]))
            else:
                f.write('\t<name>P{0}</name>\n'.format(i+1))

            if not label is None:
                f.write('\t<description>{0}</description>\n'.format(label[i]))

            zz = 0.
            if not z is None:
                zz = z[i]
            f.write('\t<Point>\n')
            f.write('\t<coordinates>{0},{1},{2}</coordinates>\n'.format(
                    x[i], y[i], zz))
            f.write('\t</Point>\n')

            f.write('</Placemark>\n')

        f.write('\n</Document>\n')
        f.write('</kml>\n')



