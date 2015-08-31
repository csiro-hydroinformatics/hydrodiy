
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message for vector size errors */
#define ESIZE 500

/* Check on uh sum */
#define UHEPS 1e-3

/* UH exponent
    daily = 2.5
    hourly = 1.25
*/
#define UHEXPON 2.5

/* Percolation factor :
   daily = 2.25
   hourly = 4
*/
#define PERCFACTOR 2.25 


/* Number of inputs required by GR4J run */
#define NINPUTS 2

/* Number of params required by GR4J run */
#define NPARAMS 4

/* Number of states returned by GR4J run */
#define NSTATES 5

/* Number of outputs returned by GR4J run */
#define NOUTPUTS 10

/* Maximim number of uh states returned by GR4J run */
#define NUH 1000

int c_gr4j_getnstates(void);

int c_gr4j_getnuh(void);

int c_gr4j_getnoutputs(void);

int c_gr4j_getesize(void);

int c_gr4j_getuh(double lag,
        int * nuh_optimised,
        double * uh);

int c_gr4j_run(int nval, int nparams, int nuh, int ninputs, 
        int nstates, int noutputs,
	double * params,
        double * uh,
	double * inputs,
        double * statesuhini,
	double * statesini,
        double * outputs);

