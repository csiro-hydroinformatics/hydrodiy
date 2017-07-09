#include "c_dscore.h"

/* comparison function for qsort ***/
static int compare(const void* p1, const void* p2)
{
    double a,b;
    a = *(double *)p1;
    b = *(double *)p2;
    if(a>b) return 1;
    if(a==b) return 0;
    if(a<b) return -1;
    return 0;
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
	double *ensemb, value, u=0, F=0, eps=1e-8;
    double sumrank;

	/* Initialisations */
    ensemb = (double*)malloc(2*ncol*sizeof(double));

    if(ensemb==NULL)
    {
        free(ensemb);
        return ENOMEM;
    }

	for(j=0;j<2*ncol;j++)
        ensemb[j] = 0.;

    /* Loop though forecast ensembles to determine */
	for(i1=0; i1<nval; i1++)
    {
	    for(i2=0; i2<nval; i2++)
        {
            if(i1==i2) continue;

            /* Get data from the two ensembles */
		    for(j=0;j<2*ncol;j++)
            {
                if(j<ncol)
			        ensemb[j] = sim[ncol*i1+j];
                else
			        ensemb[j] = sim[ncol*(i2-1)+j];
            }

            /* sort combine ensemble */
            qsort(ensemb, 2*ncol, sizeof(double), compare);

            /* Compute rank of first ensemble in combined */
            sumrank = 0;
            for(j=0; j<ncol; j++)
            {
                value = sim[ncol*i1+j];

                /* k1 and k2 are the start and end of the ensemble
                * indices where members are equal to value */
                k1 = -1;
                k2 = -1;
                for(k=0; k<2*ncol; k++)
                {
                    if(ensemb[k]>value-eps && k1<0)
                        k1 = k;

                    if(ensemb[k]>value+eps)
                    {
                        k2 = k-1;
                        break;
                    }
                }
                if(k2==-1) k2=2*ncol-1;

                sumrank += (double)(k1+k2)/2+1;
            }

            /* Comparison function as per Equation (1) in Weigel and Mason,
             * 2011 */
            fmat[i1*nval+i2] = (double)(sumrank-(ncol+1)*ncol/2)/ncol/ncol;
        }
    }

    /* Compute ranks as per Equation (2) in Weigel and Mason, 2011 */
	for(i1=0; i1<nval; i1++)
    {
        ranks[i1] = 1.;

	    for(i2=0; i2<nval; i2++)
        {
            if(i1==i2) continue;

            F = fmat[i1*nval+i2];
            u = F<0.5 ? 0. : F>0.5+eps ? 1. : 0.5;
            ranks[i1]+= u;
        }
    }

    /* closure */
    free(ensemb);

    return ierr;
}
