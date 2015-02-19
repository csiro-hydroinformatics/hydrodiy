import numpy as np
import math
import matplotlib.image as mpimg

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


def smoothpolygon(xy, tol=1e-2, maxremove=0.5):
    ''' 
    [FUNCTION IS NOT TESTED]
    Smooth a polygon defined by x/y coordinates by removing spikes, i.e.
    removing points lying far away from their neighbours.

    :param numpy.array xy : A polygon defined by a 2d numpy array [x,y]
    :param float tol : Tolerance for smooth factor. Values higher than 1e-2 allow more roughness
    :param float maxremove : Maximum proportion of points that can be removed

    '''

    # Characteristic dimension of polygon
    min = np.min(xy, 0)
    max = np.max(xy, 0)
    w = math.sqrt(np.sum((max-min)**2))

    # Initialise 
    ipb = [True]*len(xy) 
    n = len(xy)
    nmin = int(float(n)*maxremove)
    xy2 = xy.copy()

    # Remove spikes iteratively, until no spike remains
    while (np.sum(ipb)>0) & (n>nmin):
        n = len(xy2)
        d2 = np.sum((xy2[2:]-xy2[:-2])**2, 1)
        d1 = np.sum((xy2[1:]-xy2[:-1])**2, 1).reshape(n-1, 1)
        d1m = np.min(np.concatenate((d1[1:], d1[:-1]),1), 1)

        metric = (d2-d1m)/w
        metric = np.insert(metric, [0, 1], 0)
        ipb = metric > 1e-2
        xy2 = xy2[~ipb]

    return xy2







