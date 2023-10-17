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


