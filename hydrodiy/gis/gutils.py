import numpy as np
import math
import matplotlib.image as mpimg

import requests

from hydrodiy.stat import sutils

def points_inside_polygon(points, poly, rtol=1e-8, atol=1e-8):
    '''

    Determines if a set of points are inside a given polygon or not
    see [web ref from where I got the code]
    Note that points on the polygon border are not inside!

    :param numpy.array points : A list of points given as a 2d numpy array
    :param numpy.array poly : A polygon defined by a 2d numpy array [x,y]
    :param float rtol : relative tolerance for float comparison
    :param float atol : absolute tolerance for float comparison

    '''
    nvert = poly.shape[0]
    xymin = poly.min(axis=0)
    xymax = poly.max(axis=0)

    npt = points.shape[0]
    inside = np.repeat(False, npt)

    for idx in range(npt):
        x, y = points[idx,:]
        x = float(x)
        y = float(y)

        # Simple check
        if x<xymin[0] or y<xymin[1] or x>xymax[0] or y>xymax[1]:
            continue

        # Advanced algorithm
        p1x,p1y = poly[0, :]
        p1x = float(p1x)
        p1y = float(p1y)

        for i in range(nvert+1):
            p2x,p2y = poly[i % nvert, :]
            p2x = float(p2x)
            p2y = float(p2y)

            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        iseq = np.isclose(p1y, p2y, atol=atol, rtol=rtol)
                        if not iseq:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x

                        iseq = np.isclose(p1x, p2x, atol=atol, rtol=rtol)
                        if iseq or x <= xinters:
                            inside[idx] = not inside[idx]

            p1x,p1y = p2x,p2y

    return inside

def xy2kml(x, y, fkml, z=None, siteid=None, label=None):
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

    Returns
    -----------
    pwd : str
        Password

    Example
    -----------
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.linspace(0, 1, 10)
    >>> kml = gutils.xy2kml(x, y, siteid, label)
    '''

    nval = len(x)
    for v in [y, siteid, label]:
        if not v is None:
            if not len(v) == nval:
                raise ValueError('Wrong size of inputs')

    with open(fkml, 'w') as f:
        # Preamble
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://earth.google.com/kml/2.2">\n')
        f.write("<Document>\n")

        # Data
        for i in range(nval):
            f.write("\t<Placemark>\n")

            if not siteid is None:
                f.write("\t\t<name>{0}</name>\n".format(siteid[i]))

            if not label is None:
                f.write("\t\t<description>{0}</description>\n".format(label[i]))

            zz = 0.
            if not z is None:
                zz = z[i]
            f.write("\t\t<Point>\n")
            f.write("\t\t\t<coordinates>{0},{1},{2}</coordinates>\n".format(
                    x[i], y[i], zz))
            f.write("\t\t</Point>\n")

            f.write("\t</Placemark>\n")

        f.write("</Document>\n")
        f.write("</kml>\n")


def georef(name):
    ''' Extract georeference data from Google map web api

    Parameters
    -----------
    name : str
        Georeference point name

    Returns
    -----------
    info : dict
        Georeference information

    Example
    -----------
    >>> info = gutils.georef('canberra')
    '''

    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address':name}
    req = requests.get(url, params=params)

    try:
        info = req.json()
    except ValueError as err:
        raise ValueError('Cannot obtain georeference info: {0}'.format(\
            str(err)))

    if info is None:
        raise ValueError('Cannot obtain georeference info: result is None')

    info['url'] = req.url

    return info



