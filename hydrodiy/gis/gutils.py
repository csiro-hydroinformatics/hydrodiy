import numpy as np
import math
import matplotlib.image as mpimg

import time

from hydrodiy.stat import sutils

from hydrodiy import has_c_module

if has_c_module("gis", False):
    import c_hydrodiy_gis

def points_inside_polygon(points, polygon, inside=None, atol=1e-8, \
                    nprint=0):
    """

    Determines if a set of points are inside a given polygon or not
    see [web ref from where I got the code]
    Note that points on the polygon border are not inside!

    Parameters
    -----------
    points : numpy.array polygon
        Coordinates of points given as 2d numpy array [x,y]
    polygon : numpy.array polygon
        Coordinates of polygon vertices given as 2d numpy array [x,y]
    inside : numpy.array polygon
        Vector containing 0 if the point is not in the polygon or 1 for
        the opposite case.
        This input can be used to avoid allocating
        the array when calling this function multiple times.
    atol : float
        Tolerance factor for float number identity testing
    nprint : int
        Log printing frequency. No log if nprint=0

    Returns
    -----------
    inside : numpy.array
        Vector containing 0 if the point is not in the polygon or 1 for
        the opposite case.
    """
    has_c_module("gis")

    # Prepare inputs
    nprint = np.int32(nprint)
    atol = np.float64(atol)
    points = points.astype(np.float64)
    polygon = polygon.astype(np.float64)
    if inside is None:
        inside = np.zeros(len(points), dtype=np.int32)
    else:
        if not inside.dtype == np.int32:
            raise ValueError("Expected inside of "+\
                        f"dtype np.int32, got {inside.dtype}.")

        if not len(inside) == len(points):
            raise ValueError("Expected inside of length "+\
                    f"{len(points)}, got {len(inside)}.")

        # To make sure that the inside vector is properly initialised
        inside.fill(0)

    # run C code
    ierr = c_hydrodiy_gis.points_inside_polygon(atol, nprint, points, \
                    polygon, inside)

    if ierr>0:
        raise ValueError("c_hydrodiy_gis.points_inside_polygon "+\
                            "returns "+str(ierr))

    return inside


def xy2kml(x, y, fkml, z=None, siteid=None, label=None,
            icon="placemark_circle", scale=1.2):
    """ Convert a series of x/y points to KML format

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
    >>> fkml = "kml_file.kml"
    >>> gutils.xy2kml(x, y, fkml)
    """

    nval = len(x)
    for v in [y, siteid, label]:
        if not v is None:
            if not len(v) == nval:
                raise ValueError("Expected input size equal to "+\
                                    f"{nval}, got {len(v)}")
    with open(fkml, "w") as f:
        # Preamble
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n")
        f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\" "+\
            "xmlns:gx=\"http://www.google.com/kml/ext/2.2\" "+\
            "xmlns:kml=\"http://www.opengis.net/kml/2.2\" "+\
            "xmlns:atom=\"http://www.w3.org/2005/Atom\">\n\n")
        f.write("<Document>\n\n")

        # Icon
        f.write("<Style id=\"s_icon\">\n")
        f.write("\t<IconStyle>\n")
        f.write("\t\t<scale>{0:0.1f}</scale>\n".format(scale))
        f.write("\t\t<Icon>\n")
        f.write("\t\t\t<href>http://maps.google.com/mapfiles/"+\
                    f"kml/shapes/{icon}.png</href>\n")
        f.write("\t\t</Icon>\n")
        f.write("\t</IconStyle>\n")
        f.write("\t<ListStyle></ListStyle>\n")
        f.write("</Style>\n")
        f.write("<StyleMap id=\"m_icon\">\n")
        f.write("\t<Pair>\n")
        f.write("\t\t<key>normal</key>\n")
        f.write("\t\t<styleUrl>#s_icon</styleUrl>\n")
        f.write("\t</Pair>\n")
        f.write("\t<Pair>\n")
        f.write("\t\t<key>highlight</key>\n")
        f.write("\t\t<styleUrl>#s_icon</styleUrl>\n")
        f.write("\t</Pair>\n")
        f.write("</StyleMap>\n\n")

        # Data
        for i in range(nval):
            f.write("<Placemark>\n")
            f.write("\t<styleUrl>#m_icon</styleUrl>\n")

            if not siteid is None:
                f.write(f"\t<name>{siteid[i]}</name>\n")
            else:
                f.write(f"\t<name>P{i+1}</name>\n")

            if not label is None:
                f.write(f"\t<description>{label[i]}</description>\n")

            zz = 0.
            if not z is None:
                zz = z[i]
            f.write("\t<Point>\n")
            f.write(f"\t<coordinates>{x[i]},{y[i]},{zz}</coordinates>\n")
            f.write("\t</Point>\n")
            f.write("</Placemark>\n")

        f.write("\n</Document>\n")
        f.write("</kml>\n")



