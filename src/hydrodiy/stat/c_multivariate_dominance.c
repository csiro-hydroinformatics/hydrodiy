#include "c_multivariate_dominance.h"



/**************
 * Identifies points that are dominated (i.e. not part of the pareto front)
 * nval : number of points
 * ncol : number of dimensions
 * orientation : orientation of the front (-1=negative, 1=positive)
 * data : points array [nval x ncol]
 * isdominated : list of dominated points (=1). Pareto front corresponds to =0.
**/
int c_multivariate_dominance(int nval, int ncol,
    int orientation,
    int printlog,
    double* data,
    int* ndominating)
{
	int i, j, k;
    int ierr = 0;
    int i_dom_j, j_dom_i;
    double diff;

    /* Disable logging for short records (very fast) */
    int nlog = printlog > 0 ? printlog : 1;
    int toprint = (printlog > 0) && (nval >= 2 * nlog);

    if(toprint)
        fprintf(stdout, "\n\t-- Started multivariate dominance computation --\n");

	for (i = 0; i < nval; i++)
    {
        if((i % nlog == 0) && (printlog > 0))
            fprintf(stdout, "\t\tpt %8d / %8d\n", i, nval);

        /* loop over other points to check that */
        for (j = i + 1; j < nval; j++)
        {
            i_dom_j = 1; // i dominates j
            j_dom_i = 1; // j dominates i

            /* Compute pairwise difference */
            for (k = 0; k < ncol; k++)
            {
                diff = data[ncol * i + k] - data[ncol * j + k];

                /* Flip diff if orientation is not 1 */
                diff = (orientation == 1) ? diff : -diff;

                /* A nan diff means neither can dominate */
                if (isnan(diff)) {
                    i_dom_j = 0;
                    j_dom_i = 0;
                    break;
                }

                /*
                 * If diff > 0: i is better on k, so j can't dominate i on this dim.
                 * If diff < 0: j is better on k, so i can't dominate j on this dim.
                 */
                if (diff > 0) j_dom_i = 0;
                else if (diff < 0) i_dom_j = 0;

                /* Early exit if neither can possibly dominate the other */
                if (!i_dom_j && !j_dom_i) break;
            }

            ndominating[i] += i_dom_j;
            ndominating[j] += j_dom_i;
        }
    }

    return ierr;
}
