#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = ar1 x y[i] + e[i]
*   e[i] ~ N(0, sigma)
*
* nval = length of output vectors (number of values)
* seed = random generator seed
* params = algorithm parameters
*   params[0] = ar1 parameter
*   params[1] = sigma of innovation
*   params[2] = initial value 
* output = model output
*
************************************************/
int c_ar1random(int nval, double *params, 
       unsigned long int seed, double* output)
{
    const gsl_rng_type * T;
    gsl_rng * r;

	int i;
    double ar1, mu=0., sigma, y0, innov;

    /* Get parameters */
    ar1 = params[0];
    sigma = params[1];
    y0 = params[2];

    /* Check inputs */
    if((ar1<=-1)|(ar1>=1)|(sigma<0)){
        return EDOM;
    }

    /* initialise random generator */
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, seed);

    /* loop through data */
    for(i=0; i<nval; i++){
        innov = mu + gsl_ran_gaussian(r, sigma);
        y0 = ar1*y0 +innov;
        output[i] = y0;
    }
    
    gsl_rng_free(r);

    return 0;
}

/* ************** Core subroutine ******************************
* Generate ar1 process:
*   y[i+1] = ar1 x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 paramter
*   params[1] : initial value of ar1 process
* innov = innovation series e[i]
* output = model output
*
************************************************/
int c_ar1innov(int nval, double * params, 
        double * innov, double* output)
{
	int i;
    double ar1, y0;

    /* Get parameters */
    ar1 = params[0];
    y0 = params[1];

    /* loop through data */
    for(i=0; i<nval; i++){
        y0 = ar1*y0 +innov[i];
        output[i] = y0;
    }
    
    return 0;
}

/* ************** Core subroutine ******************************
* Computes innovation from ar 1 series :
*   y[i+1] = ar1 x y[i] + e[i]
*
* nval = length of output vectors (number of values)
* params = Parameters:
*   params[0] : ar1 parameter
*   params[1] : initial value of ar1 process
* input = ar1 timeseries
* innov = innovation series e[i]
*
************************************************/
int c_ar1inverse(int nval, double * params, 
        double * input, double* innov)
{
	int i;
    double ar1, y0;

    /* Get parameters */
    ar1 = params[0];
    y0 = params[1];

    /* loop through data */
    for(i=0; i<nval; i++){
        innov[i] = input[i]-ar1*y0;
        y0 = input[i];
    }
    
    return 0;
}
