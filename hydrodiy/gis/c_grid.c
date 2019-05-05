#include "c_grid.h"

double clipd(double x, double x0, double x1){
    return x<x0 ? x0 : x>x1 ? x1 : x;
}

long long clipi(long long x,long long x0, long long x1){
    return x<x0 ? x0 : x>x1 ? x1 : x;
}

long long getnxy(long long ncols, long long idxcell, long long *nxy){
    /* Returns the coordinates of cell idxcell as [icol, irow]
        (cell numbers are increasing horizontally)
    */
    nxy[0] = idxcell%ncols;
    nxy[1] = (idxcell-nxy[0])/ncols;
    return 0;
}


long long c_coord2cell(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, double * xycoords, long long * idxcell)
{
    long long ierr, i, nx, ny;
    ierr = 0;

    for(i=0; i<nval; i++)
    {
        nx = (long long)rint((xycoords[2*i]-xll)/csz);
        ny = nrows-1-(long long)rint((xycoords[2*i+1]-yll)/csz);

        if(nx<0 || nx>=ncols || ny<0 || ny>=nrows)
            idxcell[i] = -1;
        else
            idxcell[i] = ny*ncols+nx;
    }

    return ierr;
}

long long c_cell2coord(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, long long * idxcell, double * xycoords)
{
    long long ierr, i, icell;
    ierr = 0;
    double nan;
    static double zero = 0.0;

    /* nan value if not defined */
    nan = 1./zero * zero;

    for(i=0; i<nval; i++)
    {
        icell = idxcell[i];

        if(icell<0 || icell>=nrows*ncols)
        {
            xycoords[2*i] = nan;
            xycoords[2*i+1] = nan;
        }
        else
        {
            /* Compute coordinates of cell center */
            xycoords[2*i] = icell%ncols*csz+xll;
            xycoords[2*i+1] = (nrows-1-(icell-icell%ncols)/ncols)*csz+yll;
        }
    }

    return ierr;
}

long long c_neighbours(long long nrows, long long ncols,
    long long idxcell, long long * neighbours)
{
    /* Compute neigbouring cells
    *   0 1 2
    *   3 X 5
    *   6 7 8
    */
    long long ix, iy, nxy[2], nx0, nx, ny0, ny, k;

    if(idxcell<0 || idxcell>=nrows*ncols)
        return GRID_ERROR + __LINE__;

    getnxy(ncols, idxcell, nxy);
    nx0 = nxy[0];
    ny0 = nxy[1];

    for(iy=-1; iy<2; iy++){
        for(ix=-1; ix<2; ix++){

            /* Index of neighbour cell in neighbour vector */
            k = 1+ix+(1+iy)*3;

            if(ix==0 && iy==0){
                neighbours[k] = -1;
                continue;
            }

            /* Compute nx and ny for neighbour cell */
            nx = nx0 + ix;
            ny = ny0 + iy;

            /* Store neighbour cell number */
            if(nx<0 || nx > ncols-1 || ny<0 || ny>nrows-1)
                neighbours[k] = -1;
            else
                neighbours[k] = ny*ncols + nx;
        }
    }

    return 0;
}


long long c_slice(long long nrows, long long ncols,
    double xll, double yll, double csz, double* data,
    long long nval, double* xyslice, double * zslice)
{
    long long ierr, i, idxcell1[1], idxcell2[1], idxcell3[1];
    double dx, dy, xy1[2], xy2[2], xy3[2];
    double val1, val2, val3, tol;
    double nan, denom, t1, t2;
    static double zero = 0.0;

    /* nan value if not defined */
    nan = 1./zero * zero;

    /* Initialise */
    tol = 1e-10;

	/* Linear long longerpolation
	* Given 3 points (x1,y1,z1) (x2,y2,z2) (x3,y3,z3), The plan (x,y,z)
	* containing the 3 points has the following equation
	*  z 	= {  [ (z2-z1)(y3-y1) - (z3-z1)(y2-y1) ] * (x-x1)
    *                   - [(z2-z1)(x3-x1) - (z3-z1)(x2-x1)] * (y-y1)  }
				/ [(x2-x1)(y3-y1) - (y2-y1)(x3-x1)]   + z1
	*/

    for(i=0; i<nval; i++)
    {
        zslice[i] = nan;

        /* Convert to idxcell */
        ierr = c_coord2cell(nrows, ncols, xll, yll, csz, 1,
                                &(xyslice[2*i]), idxcell1);
        if(ierr > 0 || *idxcell1 < 0)
            continue;

        /* Convert to xy of nearest cell */
        ierr = c_cell2coord(nrows, ncols, xll, yll, csz, 1,
                                            idxcell1, xy1);
        if(ierr > 0 || isnan(*xy1))
            continue;

        /* Set default value */
        val1 = data[*idxcell1];
        zslice[i] = val1;

        /* Get distance from nearest cell */
        dx = xyslice[2*i]-xy1[0];
        dy = xyslice[2*i+1]-xy1[1];

        /* Compute coordinates of nearest points */
    	xy2[0] = xy1[0];
	    xy2[1] = xy1[1];
	    if(fabs(dx)>tol)
            xy2[0]=xy1[0]+dx/fabs(dx)*csz;


        xy3[0] = xy1[0];
        xy3[1] = xy1[1];
	    if(fabs(dy)>0)
            xy3[1]=xy1[1]+dy/fabs(dy)*csz;

        /* Convert nearest xy to idxcell and get value */
        ierr = c_coord2cell(nrows, ncols, xll, yll, csz, 1, xy2, idxcell2);
        if(ierr > 0 || *idxcell2 < 0)
            continue;

        ierr = c_coord2cell(nrows, ncols, xll, yll, csz, 1, xy3, idxcell3);
        if(ierr > 0 || *idxcell3 < 0)
            continue;

        /* Linear interpolation */
        val2 = data[*idxcell2];
        val3 = data[*idxcell3];

	    zslice[i] = val1;

	    if((fabs(dx)>tol && fabs(dy)<tol) || *idxcell1==*idxcell2)
        {
            zslice[i] = (val2-val1)/csz*dx+val1;
            continue;
        }

        if((fabs(dx)<tol && fabs(dy)>tol) || *idxcell1==*idxcell3)
        {
            zslice[i] = (val3-val1)/csz*dy+val1;
            continue;
        }

        if(fabs(dx)>tol && fabs(dy)>tol)
        {
	    	denom = (xy2[0]-xy1[0])*(xy3[1]-xy1[1])
                        -(xy2[1]-xy1[1])*(xy3[0]-xy1[0]);

	    	t1 = (val2-val1)*(xy3[1]-xy1[1])-(val3-val1)*(xy2[1]-xy1[1]);
	    	t2 = (val2-val1)*(xy3[0]-xy1[0])-(val3-val1)*(xy2[0]-xy1[0]);

	    	zslice[i] = (t1*dx-t2*dy)/denom + val1;
	    }
	}

    return 0;
}
