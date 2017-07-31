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
	int i1, i2, j, ierr=0, ninit;
	double value, valueprev, valuenext, index, u=0, F=0;
    double sumrank, rk, ncold, thresh, diff, diffnext;
    double nties, start, end;
    double (*ensemb)[2];

    /* check inputs */
    if(eps<1e-20)
        return EVALUE;

    if(ncol<=0 || nval <=0)
        return ESIZE;

	/* Initialisations of 2d array to store value and index */
    ensemb = malloc(sizeof *ensemb * 2*ncol);

    if(ensemb==NULL)
    {
        free(ensemb);
        return ENOMEM;
    }

    ninit = nval < 2*ncol ? 2*ncol : nval;
	for(j=0;j<ninit;j++)
    {
        if(j<2*ncol)
        {
            ensemb[j][0] = 0.;
            ensemb[j][1] = 0.;
        }

        if(j<nval) ranks[j] = 1.;
    }

    /* Maximum threshold of sum of ranks to get R==1 */
    ncold = (double) ncol;
    thresh = ncold*ncold+ncold/2;

    /* Loop though pairs of ensembles */
	for(i1=0; i1<nval; i1++)
    {
	    for(i2=i1+1; i2<nval; i2++)
        {
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
            sumrank = 0.;
            start = -1.;
            end = -1.;
            index = 0;
            nties = 0;
            valueprev = ensemb[0][0]+1.;
            valuenext = ensemb[1][0];

            /* Compute rank of first ensemble within combined */
            for(j=0; j<2*ncol; j++)
            {
                /* Get data from ensemble */
                value = ensemb[j][0];
                if(j<2*ncol-1) valuenext = ensemb[j+1][0];
                else valuenext = value+1.;
                index = ensemb[j][1];

                /* Value differences */
                diff = fabs(value-valueprev);
                diffnext = fabs(value-valuenext);

                /* Start a tie sequence */
                if(index<ncol && diff>=eps)
                {
                    start = (double) j;
                    end = (double) (j);
                    nties = 1.;
                }

                /* Continue sequence */
                if(start>=0. && diff<eps)
                {
                    /* continue a tie sequence */
                     end += 1.;

                     /* increment the number of valid ties */
                     if(index<ncol) nties += 1.;
                }

                /* End sequence */
                if(start>=0. && diffnext>=eps)
                {
                    /* Compute sumrank for the previous sequence*/
                    rk = 1. + (start+end)/2;
                    sumrank += rk*nties;

                    /* End sequence */
                    start = -1.;
                }

                /* Loop */
                valueprev = value;
            }

            /* Comparison function as per Equation (1) in Weigel and Mason,
             * 2011 */
            F = (sumrank-(ncold+1)*ncold/2)/ncold/ncold;
            fmat[i1*nval+i2] = F;

            /* Compute ranks as per Equation (2) in Weigel and Mason, 2011 */
            u = F<0.5-1e-8 ? 0. : F>0.5+1e-8 ? 1. : 0.5;
            ranks[i1] += u;
            ranks[i2] += 1.-u;
        }
    }

    /* closure */
    free(ensemb);

    return ierr;
}
