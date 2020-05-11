#include "c_grid.h"

double clipd(double x, double x0, double x1)
{
    return x<x0 ? x0 : x>x1 ? x1 : x;
}

long long clipi(long long x,long long x0, long long x1)
{
    return x<x0 ? x0 : x>x1 ? x1 : x;
}

long long getnxy(long long ncols, long long idxcell, long long *nxy)
{
    /* Returns the coordinates of cell idxcell as [icol, irow]
       Comments:
       - cell numbers are increasing from left to right first, up to
       bottom.
       - row number are increasing from bottom to top
       - col number are increasing from left to right
    */
    nxy[0] = idxcell%ncols;
    nxy[1] = (idxcell-nxy[0])/ncols;
    return 0;
}

long long getcoord(long long nrows, long long ncols, double xll, double yll,
                double csz, long long idxcell, double *coord){
    /* Returns the coordinates of cell idxcell as [x, y]
       Comments:
       - cell numbers are increasing from left to right first, up to
         bottom.
       - row number are increasing from bottom to top
       - col number are increasing from left to right
       - coordinates correspond to the centre of the cell.
    */
    long long nxy[2];
    getnxy(ncols, idxcell, nxy);
    coord[0] = xll+csz*((double)nxy[0]+0.5);
    coord[1] = yll+csz*((double)(nrows-1-nxy[1])+0.5);
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
        nx = (long long)((xycoords[2*i]-xll)/csz);
        ny = nrows-1-(long long)((xycoords[2*i+1]-yll)/csz);

        if(nx<0 || nx>=ncols || ny<0 || ny>=nrows)
            idxcell[i] = -1;
        else
            idxcell[i] = ny*ncols+nx;
    }

    return ierr;
}

long long c_cell2rowcol(long long nrows, long long ncols,
    long long nval, long long * idxcell, long long * rowcols)
{
    long long ierr, i, icell, rowcol[2];
    ierr = 0;

    for(i=0; i<nval; i++)
    {
        icell = idxcell[i];

        if(icell<0 || icell>=nrows*ncols)
        {
            rowcols[2*i] = -1;
            rowcols[2*i+1] = -1;
        }
        else
        {
            /* Compute coordinates of cell center */
            getnxy(ncols, icell, rowcol);
            rowcols[2*i+1] = rowcol[0]; // column number
            rowcols[2*i] = rowcol[1]; // row number
        }
    }

    return ierr;
}

long long c_cell2coord(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long nval, long long * idxcell, double * xycoords)
{
    long long ierr, i, icell;
    double xy[2], nan;
    static double zero = 0.0;

    /* nan value if not defined */
    nan = 1./zero * zero;

    ierr = 0;

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
            getcoord(nrows, ncols, xll, yll, csz, icell, xy);
            xycoords[2*i] = xy[0];
            xycoords[2*i+1] = xy[1];
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
    xy1[0] = 0;
    xy1[1] = 0;

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

        /* Convert to coordinate of center of nearest cell */
        ierr = getcoord(nrows, ncols, xll, yll, csz, *idxcell1, xy1);
        if(ierr > 0)
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

long long c_upstream(long long nrows, long long ncols,
    long long * flowdircode, long long * flowdir,
    long long nval, long long * idxdown, long long * idxup)
{
    /* Determines the list of upstream cell.
    Flow dir codes are organised as follows
    * 0 1 2
    * 3 X 5
    * 6 7 8
    **/

    long long i, j, k, fd, idxcell, idxneighb;
    long long neighbours[9];

    for(i=0; i<nval; i++)
    {
        /* Determines neighbouring cells */
        idxcell = idxdown[i];

        if(idxcell<0 || idxcell >= nrows*ncols)
            return GRID_ERROR + __LINE__;

        c_neighbours(nrows, ncols, idxcell, neighbours);

        /* loop through neighbours and determines if
        they drain long longo downstream cell */
        k = 0;
        for(j=0; j<9; j++){
            /* Get flow direction code of neighbours */
            idxneighb = neighbours[j];

            if(idxneighb == -1)
                continue;

            /* Skip if there is no neighbours or if
                neighbours is sink */
            fd = flowdir[idxneighb];

            if(fd==0)
                continue;

            /* Check that flow direction points towards idxcell */
            if(fd==flowdircode[8-j]){
                idxup[9*i+k] = idxneighb;
                k++;
            }
        }
        for(j=k; j<9; j++)
            idxup[9*i+j] = -1;
    }
    return 0;
}


long long c_downstream(long long nrows, long long ncols,
    long long * flowdircode, long long * flowdir,
    long long nval, long long * idxup, long long * idxdown)
{
    /* Determines the downstream cell.
    Flow dir codes are organised as follows
    * 0 1 2
    * 3 X 5
    * 6 7 8
    *
    * if the current point is a sink (ie fd=0), then idxdown = -2
    * if the current point links to outside of the grid then idxdown = -1
    **/

    long long i, j, fd, idxcell;
    long long neighbours[9];

    for(i=0; i<nval; i++)
    {
        /* Determines neighbouring cells */
        idxcell = idxup[i];

        if(idxcell<0 || idxcell >= nrows*ncols)
            return GRID_ERROR + __LINE__;

        c_neighbours(nrows, ncols, idxcell, neighbours);

        /* Flow direction */
        fd = flowdir[idxcell];

        /* Default downstream */
        idxdown[i] = -1;

        if(fd==0){
            idxdown[i] = -2;
            continue;
        }

        for(j=0; j<9; j++){
            if(fd == flowdircode[j]){
                idxdown[i] = neighbours[j];
                continue;
            }
        }
    }

    return 0;
}
long long c_accumulate(long long nrows, long long ncols,
    long long nprint, long long max_accumulated_cells,
    double nodata_to_accumulate,
    long long * flowdircode,
    long long * flowdir,
    double * to_accumulate,
    double * accumulation)
{

    long long accumulated_cells, i, ierr, ntot;
    long long idxdown[1], idxup[1];
    double accvalue;

    /* Check inputs */
    if(max_accumulated_cells < 1)
            return GRID_ERROR + __LINE__;

    if(nrows < 1 || nrows < 1)
            return GRID_ERROR + __LINE__;

    ntot = nrows*ncols;

    /* print inputs */
    fprintf(stdout, "\n\t-- Started accumulation for "
                            "grid [%lldx%lld] --\n", nrows, ncols);
    fprintf(stdout, "\tntot = %lld\n", ntot);
    fprintf(stdout, "\tnprint = %lld\n", nprint);
    fprintf(stdout, "\tmax_accumulated_cells = %lld\n",
                        max_accumulated_cells);

    for(i=0; i<ntot; i++)
    {
        if(i%nprint == 0)
            fprintf(stdout, "\t\tCompleted accumulation ... %0.1f%%\n",
                100*(double)(i)/(double)(ntot));

        /* Find downstream cell */
        idxup[0] = i;
        idxdown[0] = 0;
        accumulated_cells = 0;

        /* Loop through cells */
        while(accumulated_cells <= max_accumulated_cells)
        {
            ierr = c_downstream(nrows, ncols, flowdircode, flowdir,
                        1, idxup, idxdown);

            if(ierr>0)
                return GRID_ERROR + __LINE__;

            if(idxdown[0]<0)
            {
                accumulation[idxup[0]] = nodata_to_accumulate;
                break;
            }

            /* Get accumulated value */
            accvalue = to_accumulate[idxdown[0]];

            /* Increase flow accumulation at downstream cell */
            accumulation[idxdown[0]] += accvalue;

            /* Loop */
            accumulated_cells ++;
            idxup[0] = idxdown[0];
        }

    }

    return 0;
}


long long c_intersect(long long nrows, long long ncols,
    double xll, double yll, double csz, double csz_area,
    long long nval, double * xy_area,
    long long ncells, long long * npoints,
    long long * idxcells, double * weights)
{

    long long i, j, k, ierr, idxcell[1];
    double xy[2], areafactor;

    areafactor = (csz_area/csz)*(csz_area/csz);

    j = 0;
    for(i=0; i<nval; i++)
    {
        xy[0] = xy_area[2*i];
        xy[1] = xy_area[2*i+1];

        /* Get cell number for coordinates */
        ierr = c_coord2cell(nrows, ncols, xll, yll, csz, 1,
                xy, idxcell);

        if(ierr>0)
            continue;

        /* Look for already store cells and add weight */
        for(k=0; k<j; k++)
        {
            if(idxcells[k] == idxcell[0]){
                weights[k] += areafactor;
                break;
            }
        }

        /* If not found in existsting cells, add a new cell */
        if(k==j)
        {
            idxcells[j] = idxcell[0];
            weights[j] = areafactor;
            j++;
        }
    }
    npoints[0] = j;

    return 0;
}


long long c_voronoi(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long ncells, long long * idxcells_area,
    long long npoints, double * xypoints,
    double * weights)
{

    long long i, j, jmin, ierr, idxcell;
    double xy[2], dx, dy, dist, distmin;

    for(j=0; j<npoints; j++)
        weights[j] = 0;

    for(i=0; i<ncells; i++)
    {
        /* Get cell number for coordinates */
        idxcell = idxcells_area[i];
        xy[0] = xypoints[2*i];
        xy[1] = xypoints[2*i+1];

        ierr = getcoord(nrows, ncols, xll, yll, csz, idxcell, xy);
        if(ierr>0)
            return GRID_ERROR + __LINE__;

        /* Find closest point */
        distmin = 1e30;
        jmin = 0;
        for(j=0; j<npoints; j++){
            dx = xy[0]-xypoints[2*j];
            dy = xy[1]-xypoints[2*j+1];
            dist = sqrt(dx*dx+dy*dy);

            if(dist<distmin){
                distmin = dist;
                jmin = j;
            }
        }

        /* Store information */
        weights[jmin] += 1;
    }

    /* Convert to weights in [0, 1] */
    for(j=0; j<npoints; j++)
        weights[j]/=(double)ncells;

    return 0;
}


/*
* Computes slope in a grid
*
* Flow dir codes are organised as follows
* 0 1 2
* 3 X 5
* 6 7 8
*/
long long c_slope(long long nrows,
    long long ncols,
    long long nprint,
    double cellsize,
    long long * flowdircode,
    long long * flowdir,
    double * altitude,
    double * slopeval)
{

    long long i, ierr, ntot, fd;
    long long idxdown[1], idxup[1];
    double altup, altdown, dist;
    double sqrt2 = sqrt(2);

    /* Check inputs */
    //if(cellsize <= 1e-10)
    //        return GRID_ERROR + __LINE__;

    if(nrows < 1 || nrows < 1)
            return GRID_ERROR + __LINE__;

    ntot = nrows*ncols;

    /* Print intputs */
    fprintf(stdout, "\n\t-- Started splope calculation"
                        " for grid [%lldx%lld] --\n", nrows, ncols);
    fprintf(stdout, "\tntot = %lld\n", ntot);
    fprintf(stdout, "\tnprint = %lld\n", nprint);
    fprintf(stdout, "\tcellsize = %f\n", cellsize);

    for(i=0; i<ntot; i++)
    {
        if(i%nprint == 0)
            fprintf(stdout, "\t\tCompleted slope calculation ... %0.1f%%\n",
                100*(double)(i)/(double)(ntot));

        /* Find downstream cell */
        idxup[0] = i;
        idxdown[0] = 0;
        ierr = c_downstream(nrows, ncols, flowdircode, flowdir,
                    1, idxup, idxdown);

        if(ierr>0)
            return GRID_ERROR + __LINE__;

        /* Extract altitude of both cells */
        if(idxdown[0]>=0)
        {
            altup = altitude[i];
            altdown = altitude[idxdown[0]];

            /* Compute slope */
            fd = flowdir[i];
            dist = cellsize;
            if(fd == flowdircode[0] || fd == flowdircode[2]
                        || fd == flowdircode[6] || fd == flowdircode[8])
                dist *= sqrt2;

            slopeval[i] = (altup-altdown)/dist;
        }
    }

    return 0;
}

