#include "c_catchment.h"

/* comparison function for qsort ***/
static int compare(const void* p1, const void* p2)
{
    long long a,b;
    a = *(long long *)p1;
    b = *(long long *)p2;
    if(a>b) return 1;
    if(a==b) return 0;
    if(a<b) return -1;
    return 0;
}

long long celldist(long long nrows, long long ncols, long long n1,
                        long long n2)
{
    long long nxy1[2], nxy2[2], dx, dy;

    if(n1<0 || n1>=nrows*ncols || n2<0 || n2>=nrows*ncols)
        return CATCHMENT_ERROR + __LINE__;

    getnxy(ncols, n1, nxy1);
    getnxy(ncols, n2, nxy2);

    dx = nxy1[0]-nxy2[0];
    dx = dx<0 ? 0 : dx;

    dy = nxy1[1]-nxy2[1];
    dy = dy<0 ? 0 : dy;

    return dx>dy ? dx : dy;
}


/* Routine to delineate catchment area */
long long c_delineate_area(long long nrows, long long ncols,
    long long* flowdircode, long long * flowdir,
    long long idxoutlet,
    long long ninlets, long long * idxinlets,
    long long nval, long long * idxcells_area,
    long long * buffer1, long long * buffer2)
{
	long long i, k, l, m, idx;
    long long nbuffer1, nbuffer2, nlayer, idxcell[1];
    long long idxup[9];

    /* Check buffer size */
    if(nval<1)
        return CATCHMENT_ERROR + __LINE__;

    /* Check cell is within the grid */
    if(idxoutlet<0 || idxoutlet>nrows*ncols-1)
        return CATCHMENT_ERROR + __LINE__;

    for(m=0; m<ninlets; m++)
        if(idxinlets[m]<0 || idxinlets[m]>nrows*ncols-1)
            return CATCHMENT_ERROR + __LINE__;

    /* Initialise algorithm by including outlet cell */
    i=0;
    nbuffer2 = 1;
    buffer2[i] = idxoutlet;
    nlayer = 0;

    /* The algorithm use two buffer to identify points in the
     * catchment area:
     * - buffer1 : list of grid cells that are part of the catchment
     *              area at the current step of the algorithm.
     * - buffer2 : list of grid cells upstream of each
     */

    /* Infinite loop broken when i reaches nval */
    while(nlayer>=0)
    {
        /* Swap buffers */
        for(l=0; l<nbuffer2; l++)
            buffer1[l] = buffer2[l];

        nbuffer1 = nbuffer2;
        nbuffer2 = 0;

        /* Loop through content of first buffer (current points) */
        for(l=0; l<nbuffer1; l++)
        {
            idxcell[0] = buffer1[l];

            /* Loop around current cell */
            c_upstream(nrows, ncols, flowdircode, flowdir,
                1, idxcell, idxup);

            /* Populate second buffer (upstream points) */
            for(k=0; k<9; k++)
            {
                idx = idxup[k];
                //fprintf(stdout, " %d", idxup[k]);
                if(idx>=0)
                {
                    /* TODO check that idx is not in idxcells
                    to avoid circularity - this will make algorithm very
                    slow. A faster version would be to check circularity
                    for buffer1 only.

                    for(l=0; l<i; l++)
                        if(idx == idxcell[i])
                            return CATCHMENT_ERROR + __LINE__;
                    */

                    /* Check cell is not in the list of inlets */
                    for(m=0; m<ninlets; m++)
                    {
                        if(idxinlets[m] == idx)
                            break;
                    }

                    /* If cells are upstream of idx cell, store them */
                    if(m==ninlets)
                    {
                        /* Stop loop if we reach end of vector */
                        if(i==nval-1)
                            return CATCHMENT_ERROR + __LINE__;

                        /* Store the cell */
                        idxcells_area[i] = idx;
                        buffer2[nbuffer2] = idx;

                        /* Stop loop if we reach end of buffer */
                        if(nbuffer2==nval-1)
                            return CATCHMENT_ERROR + __LINE__;

                        nbuffer2 ++;
                        i++;
                    }
                }
            }
        }

        /* Stop algorithm if there is nothing left */
        if(nbuffer2==0)
            return 0;

        /* Add outlet in first layer */
        if(nlayer==0)
        {
            /* Check we haven't reached the maximum number of cells */
            if(i==nval-1)
                return CATCHMENT_ERROR + __LINE__;

            idxcells_area[i] = idxoutlet;
            i++;
        }

        nlayer ++;
        /*
        fprintf(stdout, "\tLayer %3d completed, found %d points\n",
                nlayer, nbuffer2);
        */
    }

    return 0;
}


long long c_delineate_boundary(long long nrows, long long ncols,
    long long nval,
    long long * idxcells_area,
    long long * buffer,
    long long * catchment_area_mask,
    long long * idxcells_boundary)
{
    long long i, k, ngrid, nbuffer, shift[4], isout;
    long long idxcell, idxcelln, distmax;
    long long next, buf, ibnd, start;
    long long nxycell[2], nxybuf[2], nxystart[2];
    long long dx, dy, dist, dmin, knext;
    double percmax;

    /* Check buffer size */
    if(nval<1)
        return CATCHMENT_ERROR + __LINE__;

    /* Grid size */
    ngrid = nrows*ncols;

    /* Maximum distance to seek for next boundary point */
    distmax = nrows > ncols ? nrows : ncols;

    /* Percentage of boundary points before computing distance from start */
    percmax = 0.8;

    /* Sort the idxcells_area */
    qsort(idxcells_area, nval, sizeof(long long), compare);

    /* Shifting of cell index  to look for neighbouring cells */
    shift[0] = -1;
    shift[1] = 1;
    shift[2] = -ncols;
    shift[3] = ncols;

    /* Step 1 - find cells on the boundary by looping through cells */
    nbuffer = 1;
    buffer[0] = idxcells_area[0];

    for(i=1; i<nval; i++)
    {
        idxcell = idxcells_area[i];

        /* Check if cell is in  area */
        if(catchment_area_mask[idxcell]!=1)
            return CATCHMENT_ERROR + __LINE__;

        /* Check if neighbouring cells are in  area */
        isout = 1;
        for(k=0; k<4; k++)
        {
            idxcelln = idxcell+shift[k];

            /* Check if neighbouring cell is in the catchment
             * area, provided it is in the grid. Otherwise
             * assumes the neighbouring cell is out of the catchment area
             * (i.e. set isout = 0) */
            if(idxcelln>=0 && idxcelln<ngrid)
                isout *= (long long)(catchment_area_mask[idxcelln] == 1);
            else
                isout = 0;
        }

        /* If at least one cell is out of the area, then
        * we are on the boundary */
        if(isout==0)
        {
            /* Check we have enough space in the buffer */
            if(nbuffer > nval)
                return CATCHMENT_ERROR + __LINE__;

            /* Store cell in the buffer */
            buffer[nbuffer] = idxcell;
            nbuffer++;
        }
    }

    /* Step 2 - reorder the cells along the boundary */
    idxcell = buffer[0];
    start = buffer[0];
    getnxy(ncols, start, nxystart);
    buffer[0] = -1;
    next = -1;
    knext = -1;

    for(ibnd=0; ibnd<nbuffer; ibnd++)
    {
        /* Cell coordinates */
        getnxy(ncols, idxcell, nxycell);

        /* Store boundary cell */
        idxcells_boundary[ibnd] = idxcell;

        /* Go through the buffer and find closest cell */
        dmin = distmax*distmax;
        for(k=0; k<nbuffer; k++)
        {
            buf = buffer[k];

            /* Skip cells already excluded */
            if(buf<0)
                continue;

            /* Cell coordinates */
            getnxy(ncols, buf, nxybuf);

            /* Compute distance */
            dx = nxycell[0]-nxybuf[0];
            dy = nxycell[1]-nxybuf[1];
            dist = dx*dx+dy*dy;

            /* Find closest cell in the buffer */
            if(dist<dmin && dist >0)
            {
                next = buf;
                knext = k;
                dmin = dist;
            }

            /* Stops if the distance is 1 (immediate neighbouring cell) */
            if(dist == 1)
                break;
        }

        /* Compute distance from start when approaching the end of the
         * boundary
         * points. This is to avoid loops */
        if(ibnd > (long long)((double)nbuffer*percmax))
        {
            dx = nxycell[0]-nxystart[0];
            dy = nxycell[1]-nxystart[1];
            dist = dx*dx+dy*dy;

            /* Break if we are closer to the start than next point */
            if(dist<dmin)
                break;
        }

        /* Iterate if we have a neighbour */
        buffer[knext] = -1;
        idxcell = next;
    }

    /* Close boundary */
    ibnd = ibnd > nval-1 ? nval-1 : ibnd;
    idxcells_boundary[ibnd] = start;

    return 0;
}

long long c_exclude_zero_area_boundary(long long nval,
    double deteps, double * xycoords, long long * idxok)
{
    long long ierr=0, i;
    double det, proj, norm, x1, y1, x2, y2, x3, y3;

    if(nval < 2)
        return CATCHMENT_ERROR + __LINE__;

    /* Set the two first points as ok */
    idxok[0] = 1;
    idxok[1] = 1;

    /* Loop through remaining points */
    for(i=2; i<nval; i++)
    {
        /* Get coordinates from three consecutive points */
        x1 = xycoords[(i-2)*2];
        y1 = xycoords[(i-2)*2+1];
        x2 = xycoords[(i-1)*2];
        y2 = xycoords[(i-1)*2+1];
        x3 = xycoords[i*2];
        y3 = xycoords[i*2+1];

        /* By default, we assume that point is ok */
        idxok[i] = 1;

        /* Check if the points are aligned
         * by computing determinant */
        det = fabs(y3*x2-y2*x3-x1*y3+x3*y1+x1*y2-x2*y1);

        /* Determinant is zero, so points are aligned */
        if(det < deteps)
        {
            /* Compute the projection of P1P2 on P1P3 */
            proj = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1);
            norm = (x3-x1)*(x3-x1)+(y3-y1)*(y3-y1);

            /* Point 2 is outside the line 1 to 3 */
            if(proj < 0 || proj > norm)
                idxok[i-1] = 0;
        }
    }

    return ierr;
}

long long c_delineate_river(long long nrows, long long ncols,
    double xll, double yll, double csz,
    long long * flowdircode,
    long long * flowdir,
    long long idxupstream,
    long long nval, long long * npoints,
    long long * idxcells,
    double * data)
{
    long long i, idxup[1], idxdown[1], ierr;
    long long nx1, ny1, nx2, ny2, ncolsdata;
    double dx, dy, dist, xy[2];

    ncolsdata = 5;

    /* Check cell */
    if(idxupstream<0 || idxupstream>nrows*ncols-1)
        return CATCHMENT_ERROR + __LINE__;

    npoints[0] = 0;
    dist = 0;
    dx = 0;
    dy = 0;

    for(i=0; i<nval; i++)
    {
        /* Store the current point */
        idxcells[i] = idxupstream;
        npoints[0] ++;

        /* find the downstream point */
        idxup[0] = idxupstream;
        c_downstream(nrows, ncols, flowdircode,
            flowdir, 1, idxup, idxdown);

        /* Distance from upstream point */
        dist += sqrt(dx*dx+dy*dy);
        data[ncolsdata*i] = dist;

        /* distance with previous point */
        data[ncolsdata*i+1] = dx;
        data[ncolsdata*i+2] = dy;

        /* coordinates */
        ierr = c_cell2coord(nrows, ncols,
            xll, yll, csz, 1, idxup,xy);
        if(ierr>0)
            return CATCHMENT_ERROR + __LINE__;

        data[ncolsdata*i+3] = xy[0];
        data[ncolsdata*i+4] = xy[1];

        /* Compute river data */
        nx1 = idxupstream%ncols;
        ny1 = (idxupstream-nx1)/ncols;

        nx2 = idxdown[0]%ncols;
        ny2 = (idxdown[0]-nx2)/ncols;

        dx = (double)(nx1-nx2);
        dy = (double)(ny1-ny2);

        /* Stop if the downstream cell is out of the grid */
        idxupstream = idxdown[0];
        if(idxupstream<0)
            return 0;
    }

    return 0;
}

long long c_delineate_flowpaths_in_catchment(long long nrows,
    long long ncols,
    long long * flowdircode,
    long long * flowdir,
    long long nval,
    long long * idxcells_area,
    long long idxcell_outlet,
    long long * flowpaths)
{
    long long ierr, i, idxcell_up[1], idxcell_down[1], ipath;

    /* Loop through all cells in catchment area */
    for(i=0; i<nval; i++)
    {
        /* initialise */
        *idxcell_up = idxcells_area[i];
        *idxcell_down = -1;
        ipath = 0;

        /* Go downstream until we reach the outlet */
        while(ipath < nval)
        {
            /* find the downstream point */
            ierr = c_downstream(nrows, ncols, flowdircode,
                        flowdir, 1, idxcell_up, idxcell_down);

            /* Break the loop if we go outside of grid limits */
            if(*idxcell_down < 0)
                break;

            /* Store downstream cell number */
            flowpaths[ipath*nval + i] = *idxcell_up;

            /* Break if we have reached the outlet cell */
            if(*idxcell_down == idxcell_outlet)
                break;

            /* Iterate */
            *idxcell_up = *idxcell_down;
            ipath++;
        }

        /* Store last cell number */
        ipath++;
        if(ipath < nval && *idxcell_down >= 0)
            flowpaths[ipath*nval + i] = *idxcell_down;
    }

    return 0;
}
