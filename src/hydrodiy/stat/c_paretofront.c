#include "c_paretofront.h"



/**************
 * Identifies points that are dominated (i.e. not part of the pareto front)
 * nval : number of points
 * ncol : number of dimensions
 * orientation : orientation of the front (-1=negative, 1=positive)
 * data : points array [nvalxncol]
 * isdominated : list of dominated points (=1). Pareto front corresponds to =0.
**/
int c_paretofront(int nval,int ncol,
    int orientation,
    double* data,
    int* isdominated)
{
	int i, j, k, dom, ierr=0;
    double diff, orientationd=(double)orientation;

	/* Computation of the mean ai and bi */
	for(i=0; i<nval; i++)
    {
        /* assume the point is not dominated by any other points */
        isdominated[i] = 0;

        /* loop over other points to check that */
        for(j=0; j<nval; j++)
        {
            /* Skip the point itself */
            if(i==j) continue;

            /* Compute pairwise difference */
            dom = 1;
            for(k=0; k<ncol; k++)
            {
                diff = data[ncol*j+k]-data[ncol*i+k];

                /* Skip if there are nan values */
                if(isnan(diff))
                    continue;

                dom *= (int)(orientationd*diff>0);
            }

            /* check if the point is dominated.
             * If yes, we stop iterating
             **/
            if(dom==1)
            {
                isdominated[i] = 1;
                break;
            }
        }
    }

    return ierr;
}
