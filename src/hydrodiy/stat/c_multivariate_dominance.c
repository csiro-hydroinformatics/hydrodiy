#include "c_multivariate_dominance.h"



/**************
 * Identifies points that are dominated (i.e. not part of the pareto front)
 * nval : number of points
 * ncol : number of dimensions
 * orientation : orientation of the front (-1=negative, 1=positive)
 * data : points array [nvalxncol]
 * isdominated : list of dominated points (=1). Pareto front corresponds to =0.
**/
int c_multivariate_dominance(int nval,int ncol,
    int orientation,
    int printlog,
    double* data,
    int* ndominating)
{
	int i, j, k, dom, ierr=0;
    int dominating;
    int count;
    double diff;
    double orientationd = (double) orientation;

    /* Disable logging for short records (very fast) */
    int nlog = 1000;
    int toprint = printlog && (nval > 2 * nlog);

    if(toprint)
        fprintf(stdout, "\n\t-- Started multivariate dominance computation --\n");

	for(i=0; i<nval; i++)
    {
        count = 0;
        if(i % nlog == 0 && toprint)
            fprintf(stdout, "\t\tpt %8d / %8d\n", i, nval);

        /* loop over other points to check that */
        for(j=0; j<nval; j++)
        {
            /* Skip the point itself */
            if(i==j) continue;

            /* Compute pairwise difference */
            dominating = 1;
            for(k=0; k<ncol; k++)
            {
                diff = data[ncol*i+k] - data[ncol*j+k];

                /* Stop if there are nan values */
                if(isnan(diff)) {
                    dominating = 0;
                    break;
                }

                /* Stop if the jth coordinate is non-dominated */
                if(diff * orientationd < 0) {
                    dominating = 0;
                    break;
                }
            }

            count += dominating;

        }

        ndominating[i] = count;

    }

    return ierr;
}
