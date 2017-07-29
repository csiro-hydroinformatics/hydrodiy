#include "c_dscore.h"

/* comparison function for qsort using 2d array */
static int compare(const void* pa, const void* pb)
{
    const double *a = pa;
    const double *b = pb;
    double eps = 1e-8;
    double diff=a[0]-b[0];

    return diff < -1*eps ? -1 : diff>eps ? 1 : 0;
}


/* ************** Core subroutine ******************************
* Calculate the discrimination score as defined by
* Weigel, Andreas P., and Simon J. Mason.
* "The generalized discrimination score for ensemble forecasts."
* Monthly Weather Review 139.9 (2011): 3069-3074.

************************************************/
int c_ensrank(double eps, int nval, int ncol, double* sim, \
        double * fmat, double * ranks)
{
	int i1, i2, j, ierr=0, debug;
	double value, valueprev, index, u=0, F=0;
    double sumrank, rk, ncold, thresh, diff;
    double nties, start, end;

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
            index = 0;
            nties = 0;
            valueprev = 1+ensemb[0][0];

            /* Compute rank of first ensemble within combined */
            for(j=0; j<2*ncol; j++)
            {
                value = ensemb[j][0];
                index = ensemb[j][1];
                diff = fabs(value-valueprev);

                if(index<ncol)
                {
                    if(diff<eps)
                    {
                        end += 1.;
                        nties += 1.;
                    }
                    else
                    {
                        /* Compute sumrank for the previous sequence*/
                        if(start>=0)
                        {
                            rk = 1. + (start+end)/2;
                            sumrank += rk*nties;

                            if(debug==1)
                                fprintf(stdout, "\t\tsumrank = %0.2f\n"
                                    "\t\t\trk = %0.2f\n"
                                    "\t\t\tdr = %0.2f = [1+(%0.0f+%0.0f)/2]x%0.0f\n",
                                        sumrank, rk,
                                        rk*nties,
                                        start, end, nties);
                        }

                        /* Initiate sequence */
                        start = (double)j;
                        end = (double)j;
                        nties = 1.;
                    }
                } else
                {
                    if(start>=0 && diff<eps && nties>1.) end += 1.;

                    /* end tie sequence */
                    if(diff>eps) nties = 0.;
                }

                /* Loop */
                valueprev = value;

                if(debug == 1)
                    fprintf(stdout, "(%d, %d) -> [%d] %0.0f:%0.2f => s=%0.0f e=%0.0f\n",
                        i1, i2, j, index, value, start, end);
            }

            /* Final sumrank computation */
            rk = 1. + (start+end)/2;
            sumrank += rk*nties;

            if(debug==1)
                fprintf(stdout, "\t\tsumrank = %0.2f\n"
                    "\t\t\trk = %0.2f\n"
                    "\t\t\tdr = %0.2f = [1+(%0.0f+%0.0f)/2]x%0.0f\n\n\n",
                        sumrank, rk,
                        rk*nties,
                        start, end, nties);

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

            u = F<0.5-1e-8 ? 0. : F>0.5+1e-8 ? 1. : 0.5;
            ranks[i1]+= u;
        }
    }

    /* closure */
    free(ensemb);

    return ierr;
}
