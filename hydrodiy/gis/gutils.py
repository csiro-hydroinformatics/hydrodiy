import numpy as np
import math
import matplotlib.image as mpimg

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

def plot_geoimage(ax, imgfile):
    ''' Plot a georeferenced image (i.e. with pngw or tifw file next to it) '''

    # Get png properties
    with open('%sw'%imgfile, 'r') as fimgw:
        # resolution
        dx = float(fimgw.readline())
        tmp = float(fimgw.readline())
        tmp = float(fimgw.readline())
        dy = float(fimgw.readline())
        # upper left corner coordiantes
        ulx = float(fimgw.readline())
        uly = float(fimgw.readline())

    img = mpimg.imread(imgfile)
    extent = [ulx, ulx+dx*img.shape[1], uly+dy*img.shape[0], uly]

    ax.imshow(img, extent=extent)


def alphashape(points, alphathresh = 10):
    ''' Alpha shape concave hull algorithm inspired by
        - http://sgillies.net/blog/1155/the-fading-shape-of-alpha/
        - http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

        CODE DOES NOT WORK!

        :param numpy.array points : A list of points given as a 2d numpy array
        :param float alphathresh: Threshold on alpha values expresed as percentile from radius distribution
    '''

    tri = Delaunay(points)
    edges = set()
    edge_points = []
    nv = len(tri.vertices)

    # Alpha-shape algorithm
    circum_r = np.zeros(nv)
    for i in range(len(tri.vertices)):
        v = tri.vertices[i]
        pa = points[v[0]]
        pb = points[v[1]]
        pc = points[v[2]]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))

        circum_r[i] = a*b*c/(4.0*area)

    # Define the alpha threshold as
    alpha = sutils.percentiles(circum_r, pthresh).squeeze()

    # Here's the radius filter.
    idx = np.where(circum_r < alpha)[0]
    v = tri.vertices[idx]

    edge_points = np.array(edge_points)

    return edge_points

