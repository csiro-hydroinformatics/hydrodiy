#include "c_dscore.h"

/* comparison function for qsort using 2d array */
static int compare(const void* pa, const void* pb)
{
    const int *a = pa;
    const int *b = pb;

    if(a[0] == b[0])
        return a[1]-b[1];
    else
        return a[0]-b[0];
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
	int i1, i2, j, k, k1, k2, ierr=0;
	double value, valueprev, index, u=0, F=0, eps=1e-8;
    double sumrank, thresh, ncold;

    double (*ensemb)[2];

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
            k1 = -1;
            k2 = -1;
            index = ensemb[j][1];
            if(index<ncol)
            {
                k1 = 0;
                k2 = 0;
            }
            valueprev = ensemb[j][0];

            fprintf(stdout, "\n\n");
            /* Compute rank of first ensemble within combined */
            for(j=0; j<2*ncol; j++)
            {
                value = ensemb[j][0];
                index = ensemb[j][1];

                if(fabs(value-valueprev)<eps && k1>=0) k2++;
                else k2 = -1;

                /* Start sequence */
                if(index<ncol && k1<0)
                {
                    k1 = j;
                    k2 = j;
                    valueprev = value;
                }

                fprintf(stdout, "[%d, %d] (%d) v=%0.1f i=%0.0f (%d %d %0.1f)\n",
                        i1, i2, j, value, index, k1, k2, valueprev);

                if(k1>=0)
                {
                    if(fabs(value-valueprev)<eps) k2++;
                    else
                    {
                        /* end sequence */
                        sumrank += 1+(double)(k1+k2)/2;
                        fprintf(stdout, "\t\tk1=%d k2=%d\n",
                            k1, k2);
                        k1 = -1;
                    }
                }
            }

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

            u = F<0.5 ? 0. : F>0.5+eps ? 1. : 0.5;
            ranks[i1]+= u;
        }
    }

    /* closure */
    free(ensemb);

    return ierr;
}
