
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Number of inputs required by dummy run */
#define DUMMY_NINPUTS 1

/* Number of params required by dummy run */
#define DUMMY_NPARAMS 1

/* Number of states returned by dummy run */
#define DUMMY_NSTATES 5

/* Number of outputs returned by dummy run */
#define DUMMY_NOUTPUTS 2

int c_dummy_getnstates(void);

int c_dummy_getnoutputs(void);

int dummy_runtimestep(int nparams, 
    int ninputs,
    int nstates, 
    int noutputs,
	double * params,
    double * inputs,
	double * states,
    double * outputs);

int c_dummy_run(int nval, 
    int nparams, 
    int ninputs, 
    int nstates, 
    int noutputs,
	double * params,
	double * inputs,
	double * statesini,
    double * outputs);

