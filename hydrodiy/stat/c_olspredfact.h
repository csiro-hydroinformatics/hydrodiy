#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


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
        double * tXXinv, double* prediction_factors);

