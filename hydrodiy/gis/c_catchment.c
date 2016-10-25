#include "c_catchment.h"

/* comparison function for qsort ***/
static int longcompare(const void* p1, const void* p2){
    long a,b;
    a = *(long *)p1;
    b = *(long *)p2;
    if(a>b) return 1;
    if(a==b) return 0;
    if(a<b) return -1;
    return 0;
}

long celldist(long nrows, long ncols, long n1, long n2){
    long nxy1[2], nxy2[2], dx, dy;

    if(n1<0 || n1>=nrows*ncols || n2<0 || n2>=nrows*ncols)
        return CATCHMENT_ERROR + __LINE__;

    getnxy(ncols, n1, nxy1);
    getnxy(ncols, n2, nxy2);

    dx = abs(nxy1[0]-nxy2[0]);
    dy = abs(nxy1[1]-nxy2[1]);

    return dx>dy ? dx : dy;
}


long c_upstream(long nrows, long ncols,
    long * flowdircode, long * flowdir,
    long nval, long * idxdown, long * idxup){

    /* Determines the list of upstream cell.
    Flow dir codes are organised as follows
    * 0 1 2
    * 3 X 5
    * 6 7 8
    **/

    long i, j, k, fd, idxcell, idxneighb;
    long neighbours[9];

    for(i=0; i<nval; i++){
        /* Determines neighbouring cells */
        idxcell = idxdown[i];

        if(idxcell<0 || idxcell >= nrows*ncols)
            return CATCHMENT_ERROR + __LINE__;

        c_neighbours(nrows, ncols, idxcell, neighbours);

        /* loop through neighbours and determines if
        they drain longo downstream cell */
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


long c_downstream(long nrows, long ncols,
    long * flowdircode, long * flowdir,
    long nval, long * idxup, long * idxdown){

    /* Determines the downstream cell.
    Flow dir codes are organised as follows
    * 0 1 2
    * 3 X 5
    * 6 7 8
    *
    * if the current point is a sink (ie fd=0), then idxdown = -2
    * if the current point links to outside of the grid then idxdown = -1
    **/

    long i, j, fd, idxcell;
    long neighbours[9];

    for(i=0; i<nval; i++){
        /* Determines neighbouring cells */
        idxcell = idxup[i];

        if(idxcell<0 || idxcell >= nrows*ncols)
            return CATCHMENT_ERROR + __LINE__;

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


long c_delineate_area(long nrows, long ncols,
    long* flowdircode, long * flowdir,
    long idxoutlet,
    long ninlets, long * idxinlets,
    long nval, long * idxcells,
    long * buffer1, long * buffer2)
{
	long i, k, l, m, idx;
    long nbuffer1, nbuffer2, nlayer, idxcell[1];
    long idxup[9];

    /* Check buffer size */
    if(nval<1)
        return CATCHMENT_ERROR + __LINE__;

    /* Check cell */
    if(idxoutlet<0 || idxoutlet>nrows*ncols-1)
        return CATCHMENT_ERROR + __LINE__;

    for(m=0; m<ninlets; m++)
        if(idxinlets[m]<0 || idxinlets[m]>nrows*ncols-1)
            return CATCHMENT_ERROR + __LINE__;

    /* Initialise algorithm with outlet cell */
    i=0;
    nbuffer2 = 1;
    buffer2[i] = idxoutlet;
    nlayer = 0;

    /* Infinite loop broken when i reaches nval */
    while(nlayer>=0){

        /* Swap buffers */
        for(l=0; l<nbuffer2; l++)
            buffer1[l] = buffer2[l];

        nbuffer1 = nbuffer2;
        nbuffer2 = 0;

        /* Loop through content of first buffer (current points) */
        for(l=0; l<nbuffer1; l++){

            idxcell[0] = buffer1[l];

            /* Loop around current cell */
            c_upstream(nrows, ncols, flowdircode, flowdir,
                1, idxcell, idxup);

            /* Populate second buffer (upstream points) */
            for(k=0; k<9; k++){
                idx = idxup[k];
                //fprintf(stdout, " %d", idxup[k]);
                if(idx>=0){

                    /* TODO check that idx is not in idxcells
                    to avoid circularity - this will make algorithm very slow
                    A faster version would be to check circularity for buffer1
                    only

                    for(l=0; l<i; l++)
                        if(idx == idxcell[i])
                            return CATCHMENT_ERROR + __LINE__;
                    */

                    /* Check cell is not in the list of inlets */
                    for(m=0; m<ninlets; m++){
                        if(idxinlets[m] == idx)
                            break;
                    }

                    /* If cells are upstream of idx cell, store them */
                    if(m==ninlets){
                        idxcells[i] = idx;
                        buffer2[nbuffer2] = idx;

                        /* Stop loop if we reach end of vector */
                        if(i==nval-1)
                            return CATCHMENT_ERROR + __LINE__;

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
        if(nlayer==0){
            idxcells[i] = idxoutlet;

            if(i==nval-1)
                return CATCHMENT_ERROR + __LINE__;

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


long c_delineate_boundary(long nrows, long ncols,
    long nval,
    long * idxcells_area,
    long * buffer,
    long * grid_area,
    long * idxcells_boundary)
{
    long i, k, ngrid, nbuffer, shift[4], isout;
    long idxcell, idxcelln, distmax;
    long next, buf, ibnd, start;
    long nxycell[2], nxybuf[2], nxystart[2];
    long dx, dy, dist, dmin, knext;
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
    qsort(idxcells_area, nval, sizeof(long), longcompare);

    /* Shifting of cell index  to look for neighbouring cells */
    shift[0] = -1;
    shift[1] = 1;
    shift[2] = -ncols;
    shift[3] = ncols;

    /* Step 1 - find cells on the boundary by looping through cells */
    nbuffer = 1;
    buffer[0] = idxcells_area[0];

    for(i=1; i<nval; i++){

        idxcell = idxcells_area[i];

        /* Check if cell is in  area */
        if(grid_area[idxcell]!=1)
            return CATCHMENT_ERROR + __LINE__;

        /* Check if neighbouring cells are in  area */
        isout = 1;
        for(k=0; k<4; k++)
        {
            idxcelln = idxcell+shift[k];
            if(idxcelln>=0 && idxcelln<ngrid)
                isout *= (long)(grid_area[idxcelln] == 1);
        }

        if(isout==0)
        {
            buffer[nbuffer] = idxcell;
            nbuffer++;
        }

    }

    /* Step 2 - reorder the boundary cells along the boundary */
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

            /* Find closest cell */
            if(dist<dmin)
            {
                next = buf;
                knext = k;
                dmin = dist;
            }

            /* Stops if the distance is 1 (immediate neighbouring cell) */
            if(dist == 1)
                break;
        }

        /* Compute distance from start  when approacing the end of the boundary
         * points */
        if(ibnd > (long)((double)nbuffer*percmax))
        {
            dx = nxycell[0]-nxystart[0];
            dy = nxycell[1]-nxystart[1];
            dist = dx*dx+dy*dy;

            /* Break if we are closer to the start than next point */
            if(dist<dmin)
                break;
        }

        /* Iterate if we have a neighbour, else return */
        buffer[knext] = -1;
        idxcell = next;

    }

    /* Close boundary */
    ibnd = ibnd > nval-1 ? nval-1 : ibnd;
    idxcells_boundary[ibnd] = start;

    return 0;
}


long c_delineate_river(long nrows, long ncols,
    double xll, double yll, double csz,
    long * flowdircode,
    long * flowdir,
    long idxupstream,
    long nval, long * npoints,
    long * idxcells,
    double * data){

    long i, idxup[1], idxdown[1], ierr;
    long nx1, ny1, nx2, ny2, ncolsdata;
    double dx, dy, dist, xy[2];

    ncolsdata = 5;

    /* Check cell */
    if(idxupstream<0 || idxupstream>nrows*ncols-1)
        return CATCHMENT_ERROR + __LINE__;

    npoints[0] = 0;
    dist = 0;
    dx = 0;
    dy = 0;

    for(i=0; i<nval; i++){
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


long c_accumulate(long nrows, long ncols,
    long nprint, long maxarea,
    long * flowdircode,
    long * flowdir,
    long * accumulation){

    long area, i, ierr, ntot;
    long idxdown[1], idxup[1];

    ntot = nrows*ncols;

    fprintf(stdout, "\n\t-- Started accumulation for grid [%ldx%ld] --\n", nrows, ncols);
    fprintf(stdout, "\tntot = %ld\n", ntot);
    fprintf(stdout, "\tnprint = %ld\n", nprint);
    fprintf(stdout, "\tmaxarea = %ld\n", maxarea);

    for(i=0; i<ntot; i++){

        if(i%nprint == 0)
            fprintf(stdout, "\t\tCompleted accumulation ... %0.1f%%\n",
                100*(double)(i)/(double)(ntot));

        /* Find downstream cell */
        idxup[0] = i;
        idxdown[0] = 0;
        area = 0;

        while(area <= maxarea){

            ierr = c_downstream(nrows, ncols, flowdircode, flowdir,
                        1, idxup, idxdown);

            if(ierr>0)
                return CATCHMENT_ERROR + __LINE__;

            if(idxdown[0]<0)
                break;

            /* Increase flow accumulation at downstream cell */
            accumulation[idxdown[0]] +=1;

            /* Loop */
            area ++;
            idxup[0] = idxdown[0];
        }

    }

    return 0;
}


long c_intersect(long nrows, long ncols,
    double xll, double yll, double csz, double csz_area,
    long nval, double * xy_area,
    long ncells, long * npoints,
    long * idxcells, double * weights){

    long i, j, k, ierr, idxcell[1];
    double xy[2], areafactor;

    areafactor = (csz_area/csz)*(csz_area/csz);

    j = 0;
    for(i=0; i<nval; i++){
        xy[0] = xy_area[2*i];
        xy[1] = xy_area[2*i+1];

        /* Get cell number for coordinates */
        ierr = c_coord2cell(nrows, ncols, xll, yll, csz, 1,
                xy, idxcell);

        if(ierr>0)
            continue;

        /* Look for already store cells and add weight */
        for(k=0; k<j; k++){
            if(idxcells[k] == idxcell[0]){
                weights[k] += areafactor;
                break;
            }
        }

        /* If not found in existsting cells, add a new cell */
        if(k==j){
            idxcells[j] = idxcell[0];
            weights[j] = areafactor;
            j++;
        }
    }
    npoints[0] = j;



    return 0;
}

long c_voronoi(long nrows, long ncols,
    double xll, double yll, double csz,
    long ncells, long * idxcells_area,
    long npoints, double * xypoints,
    double * weights){

    long i, j, jmin, ierr, idxcell[1];
    double xy[2], dx, dy, dist, distmin;

    for(j=0; j<npoints; j++)
        weights[j] = 0;

    for(i=0; i<ncells; i++){

        /* Get cell number for coordinates */
        idxcell[0] = idxcells_area[i];
        xy[0] = xypoints[2*i];
        xy[1] = xypoints[2*i+1];

        ierr = c_cell2coord(nrows, ncols, xll, yll, csz, 1,
                idxcell, xy);

        if(ierr>0)
            return CATCHMENT_ERROR + __LINE__;

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
