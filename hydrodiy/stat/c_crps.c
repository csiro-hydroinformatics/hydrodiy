#include "c_crps.h"

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
* Calculate the CRPS as defined by
* Hersbach, H. Decomposition of the continuous ranked probability score for ensemble prediction systems Weather and Forecasting, 2000, 15, 559-570
*
* nval = length of input vectors (number of values)
* ncol = nb of ensembles
* use_weights = use weight vector or assumes weight=1/nval
* is_sorted = are ensemble already sorted ?
* obs = vector of observed data (nval x 1)
* sim = matrix of predictions (nval x ncol)
* W = Weights used to compute the average (nval x 1)
* reliability_table = matrix containing the CRPS values for each bin (ncol+1 x 7)
*	reliability_table(:,1) = pi (=i/ncol) i=0..ncol
*	reliability_table(:,2) = ai (Eq 26,27,29 from Hersbach, 2000)
*	reliability_table(:,3) = bi (Eq 26,27,29 from Hersbach, 2000)
*	reliability_table(:,4) = gi (Eq 30 from Hersbach, 2000)
*	reliability_table(:,5) = oi (Eq 31 from Hersbach, 2000)
*	reliability_table(:,6) = Reli_i (Eq 36 from Hersbach, 2000)
*	reliability_table(:,7) = crps_potential_i (Eq 37 from Hersbach, 2000)
*
* crps_decompos = average values of the CRPS
*	crps_decompos(0) = CRPS (Eq 28 from Hersbach, 2000)
*	crps_decompos(1) = Reli (Eq 36 from Hersbach, 2000)
*	crps_decompos(2) = Resol (Eq 38 from Hersbach, 2000)
*	crps_decompos(3) = uncertainty (Eq 20 from Hersbach, 2000)
*	crps_decompos(4) = CRPS potential (when predictions are corrected to get perfect reliability)
*
************************************************/
int c_crps(int nval,int ncol,
    int use_weights, int is_sorted,
    double* obs, double* sim,
    double* weights_vector,
    double* reliability_table,
    double* crps_decompos)
{
	int i, j, k, ncol_rt=7;
	double *ensemb, weight, weight_k;
    double crps_potential, pj, uncertainty, delta_obs;
	double *a, *b, *g, *o, *r, *c;

	/* Initialisations */
    ensemb = (double*)malloc((ncol+1)*sizeof(double));
    a = (double*)malloc((ncol+1)*sizeof(double));
    b = (double*)malloc((ncol+1)*sizeof(double));
    g = (double*)malloc((ncol+1)*sizeof(double));
    o = (double*)malloc((ncol+1)*sizeof(double));
    r = (double*)malloc((ncol+1)*sizeof(double));
    c = (double*)malloc((ncol+1)*sizeof(double));

    if( (ensemb==NULL)|(a==NULL)|
            (b==NULL)|(g==NULL)|
            (o==NULL)|(r==NULL)|
            (c==NULL))
    {
        free(ensemb);
        free(a);
        free(b);
        free(g);
        free(o);
        free(r);
        free(c);
        return ENOMEM;
    }

    crps_potential = 0.0;
    uncertainty = 0.0;

	for(j=0;j<ncol+1;j++)
    {
        a[j]=0;
        b[j]=0;
        g[j]=0;
        o[j]=0;
    }

	/* Computation of the mean ai and bi */
	for(i=0; i<nval; i++)
    {
        /* get predicted ensemble (columnwise)
         * we could get rid of this and work on sim,
         * but I prefer copying data before playing with it */
		for(j=0;j<ncol;j++)
			ensemb[j] = sim[ncol*i+j];

        /* sort ensemble if required */
        if(is_sorted==0)
            qsort(ensemb, ncol, sizeof(double), compare);

        /* point weight */
        weight = 1/(double)(nval);
        if(use_weights==1)
            weight = weights_vector[i];

		/* Computing the ai and bi (Eq 26) */
		for(j=0;j<ncol-1;j++)
        {
            /* Intercept problem with sorting */
            if(ensemb[j+1]<ensemb[j])
            {
                free(ensemb);
                free(a);
                free(b);
                free(g);
                free(o);
                free(r);
                free(c);
                return EDOM;
            }

            /* compute CRPS decomposition variables */
			if(obs[i]<=ensemb[j])
                b[j+1] += (ensemb[j+1]-ensemb[j]) * weight;

			if(obs[i]>=ensemb[j+1])
                a[j+1] += (ensemb[j+1]-ensemb[j]) * weight;

			if((obs[i]>ensemb[j]) && (obs[i]<ensemb[j+1]))
            {
 				a[j+1] += (obs[i]-ensemb[j]) * weight;
 				b[j+1] += (ensemb[j+1]-obs[i]) * weight;
			}
		}

		/* b0 (Eq 27) */
		if(obs[i]<ensemb[0])
            b[0] += (ensemb[0]-obs[i]) * weight;

		/* aN (Eq 27) */
		if(obs[i]>=ensemb[ncol-1])
            a[ncol] += (obs[i]-ensemb[ncol-1]) * weight;

        /* o0 (Eq 33) */
		if(obs[i]<ensemb[0])
            o[0] += weight;

		/* oN (Eq 33) */
		if(obs[i]<ensemb[ncol-1])
            o[ncol] += weight;

		/* Computation of uncertainty (Eq 19) */
		for(k=0;k<i;k++)
        {
            delta_obs =  fabs(obs[k]-obs[i]);

            weight_k = 1/(double)(nval);
            if(use_weights==1)
                weight_k = weights_vector[k];

            uncertainty += weight * weight_k * delta_obs;
		}
	}

	/* Computation of the oi, gi, Reli_i and crps_potential_i
     * from the ai and bi (Eq 30, 31, 33, 36 and 37) */
	for(j=0;j<ncol+1;j++)
    {
		/* Probability */
		pj = (double)j/(double)(ncol);

		/* Outliers (Eq 33) */
		if((j==0) && (o[j]!=0.0))
            g[j] = b[j]/o[j];

		if((j== ncol) && (o[j]!=1.0))
            g[j] = a[j]/(1-o[j]);

		/* Other cases */
		if((j>0) && (j<ncol))
        {
			g[j] = a[j]+b[j];
			o[j] = b[j]/g[j];
		}

  	    /* Eq 36 */
		r[j] = g[j]*pow(o[j]-pj,2);

        /* weights_vector[i]/sum_weightsEq 37 */
		c[j] = g[j]*o[j]*(1-o[j]);

        /* store data in the reliability_table matrix */
		reliability_table[j*ncol_rt] = pj;
		reliability_table[j*ncol_rt+1] = a[j];
		reliability_table[j*ncol_rt+2] = b[j];
		reliability_table[j*ncol_rt+3] = g[j];
		reliability_table[j*ncol_rt+4] = o[j];
		reliability_table[j*ncol_rt+5] = r[j];
		reliability_table[j*ncol_rt+6] = c[j];

        /* CRPS value */
		crps_decompos[0] += a[j]*pow(pj,2)+b[j]*pow(1-pj,2);
		if(g[j]>0)
        {
            /* CRPS Reliability */
			crps_decompos[1] += r[j];
			crps_potential += c[j];
		}
	}

	/* Final computation of Resolution */
	crps_decompos[2] = uncertainty-crps_potential;
	crps_decompos[3] = uncertainty;
	crps_decompos[4] = crps_potential;

    /* closure */
    free(ensemb);
    free(a);
    free(b);
    free(g);
    free(o);
    free(r);
    free(c);

    return 0;
}
