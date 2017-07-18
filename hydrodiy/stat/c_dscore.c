#include "c_dscore.h"

/* comparison function for qsort using 2d array */
static int compare(const void* pa, const void* pb)
{
    const double *a = pa;
    const double *b = pb;
    double diff=a[0]-b[0];

    return diff<-DSCORE_EPS ? -1 : diff>DSCORE_EPS ? 1 : 0;
}


/* ************** Core subroutine ******************************
* Calculate the discrimination score as defined by
* Weigel, Andreas P., and Simon J. Mason.
* "The generalized discrimination score for ensemble forecasts."
* Monthly Weather Review 139.9 (2011): 3069-3074.

************************************************/
int c_ensrank(int nval, int ncol, double* sim, \
        double * fmat, double * ranks)
{
	int i1, i2, j, start, end, ierr=0, debug;
	double value, valueprev, index, u=0, F=0;
    double sumrank, ncold, thresh, diff;

    double (*ensemb)[2];

    debug = 1;

	/* Initialisations of 2d array to store value and index */
    ensemb = malloc(sizeof *ensemb * 2*ncol);

    if(ensemb==NULL)
    {
        free(ensemb);
        return ENOMEM;
    }

	for(j=0;j<2*ncol;j++)
    {
        ensemb[j][0] = 0.;
        ensemb[j][1] = 0.;
    }

    /* Maximum threshold of sum of ranks to get R==1 */
    ncold = (double) ncol;
    thresh = ncold*ncold+ncold/2;

    /* Loop though pairs of ensembles */
	for(i1=0; i1<nval; i1++)
    {
	    for(i2=i1+1; i2<nval; i2++)
        {

            /* Comparison function for deterministic ensembles */
            if(ncol == 0)
            {
                sumrank = sim[i1]<sim[i2] ? 1 : 2;
                fmat[i1*nval+i2] = (sumrank-(ncold+1)*ncold/2)/ncold/ncold;
                continue;
            }

            /* Get data and indexes from the two ensembles */
		    for(j=0;j<2*ncol;j++)
            {
                if(j<ncol)
			        ensemb[j][0] = sim[ncol*i1+j];
                else
			        ensemb[j][0] = sim[ncol*(i2-1)+j];

                ensemb[j][1] = (double) j;
            }

            /* sort combined ensemble */
            qsort(ensemb, 2*ncol, sizeof ensemb[0], compare);

            /* Initialise */
            sumrank = 0;
            start = -1;
            end = -1;
            index = ensemb[j][1];
            valueprev = ensemb[j][0];

            /* Compute rank of first ensemble within combined */
            for(j=0; j<2*ncol; j++)
            {
                value = ensemb[j][0];
                index = ensemb[j][1];
                diff = fabs(value-valueprev);

                if(index<ncol)
                {
                    if(diff<DSCORE_EPS) end++;
                    else
                    {
                        /* Compute sumrank for the previous sequence*/
                        if(start>=0)
                        {
                            sumrank += 1. + (double)(start+end)/2;

                            if(debug==1)
                                fprintf(stdout, "\t\tsumrank = %0.2f (+%0.2f)\n",
                                    sumrank, (double)(start+end)/2);
                        }

                        /* Initiate sequence */
                        start = j;
                        end = j;
                    }
                } else
                    if(start>=0 && diff<DSCORE_EPS) end++;

                /* Loop */
                valueprev = value;

                if(debug == 1)
                    fprintf(stdout, "(%d, %d) -> [%d] %0.0f:%0.2f => s=%d e=%d\n",
                        i1, i2, j, index, value, start, end);
            }

            /* Final sumrank computation if the last point is not in a sequence */
            sumrank += 1. + (double)(start+end)/2;

            if(debug == 1)
                fprintf(stdout, "\t\tFinal sumrank = %0.2f (+%0.2f)\n\n",
                                    sumrank, (double)(start+end)/2);

            /* Comparison function as per Equation (1) in Weigel and Mason,
             * 2011 */
            fmat[i1*nval+i2] = (sumrank-(ncold+1)*ncold/2)/ncold/ncold;
        }
    }

    /* Compute ranks as per Equation (2) in Weigel and Mason, 2011 */
	for(i1=0; i1<nval; i1++)
    {
        ranks[i1] = 0.;

	    for(i2=0; i2<nval; i2++)
        {
            if(i1==i2) continue;

            if(i2>i1) F = fmat[i1*nval+i2];
            else F = 1-fmat[i2*nval+i1];

            u = F<0.5 ? 0. : F>0.5+DSCORE_EPS ? 1. : 0.5;
            ranks[i1]+= u;
        }
    }

    /* closure */
    free(ensemb);

    return ierr;
}
