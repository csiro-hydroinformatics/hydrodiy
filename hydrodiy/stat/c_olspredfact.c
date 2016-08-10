#include "c_olspredfact.h"


/* ************** Core subroutine ******************************
* Compute the prediction factor based on Equation 3.48 in
* Johnston and Di Nardo, Econometric Methods
*
* For a set of predictand X in R^(p, 1), the factor is
* factor = sqrt(1+ X' (X0'X0)^-1 X)
*
* nval = Numberof samples
* npreds = Nunber of predictors (i.e. p)
* predictors = Predictor matrix in R^(n, p)
* tXXinv = Inverse of the the product (X0' x X0)
*
************************************************/
int c_olspredfact(int nval, int npreds, double * predictors,
        double * tXXinv, double* prediction_factors)
{
	int i, j, k;
    double a, b;

    /* loop through data */
    for(i=0; i<nval; i++){

        prediction_factors[i] = 0;

        for(j=0; j<npreds; j++)
            for(k=0; k<npreds; k++)
                {
                    a = tXXinv[npreds*k + j];
                    b = predictors[nval*j+i]*predictors[nval*k+i]*a;
                    prediction_factors[i] += a*b;
                }
    }

    return 0;
}

