
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Number of inputs required by LAG ROUTE run */
#define LAGROUTE_NINPUTS 5

/* Number of params required by LAG ROUTE run */
#define LAGROUTE_NPARAMS 5

/* Number of states returned by LAG ROUTE run */
#define LAGROUTE_NSTATES 5

/* Number of outputs returned by LAG ROUTE run */
#define LAGROUTE_NOUTPUTS 10

int c_lagroute_getnstates(void);

int c_lagroute_getnoutputs(void);

int c_lagroute_run(int nval, 
        int nparams, 
        int nuh, 
        int ninputs, 
        int nconfig, 
        int nstates, 
        int noutputs,
        double * config,
	double * params,
        double * uh,
	double * inputs,
        double * statesuh,
	double * states,
        double * outputs);

