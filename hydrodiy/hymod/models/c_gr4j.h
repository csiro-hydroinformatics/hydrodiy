
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Percolation factor :
   daily = 2.25
   hourly = 4
*/
#define GR4J_PERCFACTOR 2.25 


/* Number of inputs required by GR4J run */
#define GR4J_NINPUTS 5

/* Number of params required by GR4J run */
#define GR4J_NPARAMS 4

/* Number of states returned by GR4J run */
#define GR4J_NSTATES 5

/* Number of outputs returned by GR4J run */
#define GR4J_NOUTPUTS 20

int gr4j_production(double P, double E, 
        double S, 
        double state0, 
        double * prod);

int gr4j_runtimestep(int nparams, 
    int nuh1, int nuh2, int ninputs,
    int nstates, int noutputs,
	double * params,
    double * uh1,
    double * uh2,
    double * inputs,
	double * statesuh,
    double * states,
    double * outputs);

int c_gr4j_run(int nval, 
    int nparams, 
    int nuh1, 
    int nuh2,  
    int ninputs, 
    int nstates, 
    int noutputs,
	double * params,
    double * uh1,
    double * uh2,
	double * inputs,
    double * statesuhini,
	double * statesini,
    double * outputs);

