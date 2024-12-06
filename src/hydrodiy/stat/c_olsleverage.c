#include "c_olsleverage.h"

/* ************** Core subroutine ******************************
* Compute the OLS leverage based on Equation 3.48 in
* Johnston and Di Nardo, Econometric Methods
*
* For a set of predictand X in R^(p, 1), the factor is
* leverage = diag(X' (X0'X0)^-1 X))
*
* nval = Numberof samples
* npreds = Nunber of predictors (i.e. p)
* predictors = Predictor matrix in R^(n, p)
* tXXinv = Inverse of the the product (X0' x X0)
*
************************************************/
int c_olsleverage(int nval, int npreds, double * predictors,
        double * tXXinv, double* leverage)
{
	int i, j, k;
    double pred1, pred2, xx, lev;

    /* loop through data */
    for(i=0; i<nval; i++)
    {
        lev = 0;

        for(j=0; j<npreds; j++)
        {
            pred1 = predictors[npreds*i+j];

            for(k=0; k<npreds; k++)
            {
                xx = tXXinv[npreds*j + k];
                pred2 = predictors[npreds*i+k];
                lev += pred1*pred2*xx;
            }
            leverage[i] = lev;
        }
    }

    return 0;
}

