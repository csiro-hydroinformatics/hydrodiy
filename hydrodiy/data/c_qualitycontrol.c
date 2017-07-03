#include "c_qualitycontrol.h"

/* Detect linear streches */
int c_islin(int nval, double thresh, double tol, int npoints,
    double * data, int * islin)
{
    int ierr=0, i, k, count, start, lintype;
    double dist, vprec, vnext, vcur;

    /* initialisation */
    vprec = data[0];
    vcur = data[1];
    count = 0;
    start = 0;

    islin[0] = 0;
    islin[1] = 0;

    /* loop through data */
    for(i=2; i<nval; i++)
    {
        vnext = data[i];
        dist = fabs(vcur-(vprec+vnext)/2);
        islin[i] = 0;

        /* check linearity if vcur is freater than thresh */
        if(dist<tol && vcur>thresh)
        {
            /* Start a new event
            * (i-2 and not i because vcur corresponds to i-1,
            * so we start the event at i-2 to include the preceeding
            * point as well)
            */
            if(count == 0) start = i-2;

            /* increment event duration */
            count += 1;

            /* Define linear flag */
            lintype = 1;
            if(fabs(vnext-vprec)<tol) lintype = 2;
        }
        else
        {
            /* End an event and set islin if event is long enough */
            if(count>=npoints)
                for(k=start; k<i; k++) islin[k] = lintype;

            /* Reset */
            count = 0;
        }

        /* loop */
        vprec = vcur;
        vcur = vnext;
    }

    return ierr;
}

