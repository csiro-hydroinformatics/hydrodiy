#ifndef __HY_STAT_OLSLEVERAGE__
#define __HY_STAT_OLSLEVERAGE__


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


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
        double * tXXinv, double* leverage);

#endif
