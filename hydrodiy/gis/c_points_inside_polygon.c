#include "c_points_inside_polygon.h"

int c_inside(int nprint, int npoints, double * points,
    int nvertices, double * polygon,
    double atol,
    double * polygon_xlim, double * polygon_ylim,
    int * inside)
{
    int ipt, ivert, k;
    double x, y, p1x, p1y, p2x, p2y;
    double xinters, dist;

    /* Print intputs */
    if(nprint > 0)
    {
        fprintf(stdout, "\n\t-- Started inside calculation --\n");
        fprintf(stdout, "\tnpoints   = %d\n", npoints);
        fprintf(stdout, "\tnvertices = %d\n", nvertices);
    }

    for(ipt=0; ipt<npoints; ipt++)
    {
        if(nprint > 0)
            if(ipt%nprint== 0 && ipt > 0)
                fprintf(stdout,
                    "\t\tInside calculation running ... %0.1f%%\n",
                        100*(double)(ipt)/(double)(npoints));

        x = points[2*ipt];
        y = points[2*ipt+1];

        /* Simple check if the point is outside polygon area */
        if(x < polygon_xlim[0] || x > polygon_xlim[1] ||
                y < polygon_ylim[0] || y > polygon_ylim[1])
            continue;

        /* Point inside polygon algorithm */
        p1x = polygon[0];
        p1y = polygon[1];
        inside[ipt] = 0;

        for(ivert=1; ivert<nvertices+1; ivert++)
        {
            k = 2*(ivert % nvertices);
            p2x = polygon[k];
            p2y = polygon[k+1];

            if(y > fmin(p1y, p2y))
            {
                if(y <= fmax(p1y, p2y))
                {
                    if(x <= fmax(p1x, p2x))
                    {
                        dist = fabs(p1y-p2y);
                        xinters = p1x;
                        if(dist > atol)
                            xinters += (y-p1y)*(p2x-p1x)/(p2y-p1y);

                        dist = fabs(p1x-p2x);
                        if(dist < atol || x <= xinters)
                            inside[ipt] = 1-inside[ipt];
                    }
                }
            }
            p1x = p2x;
            p1y = p2y;
        }

    }
    if(nprint > 0)
        fprintf(stdout, "\t-- Completed inside calculation --\n\n");

    return 0;
}

