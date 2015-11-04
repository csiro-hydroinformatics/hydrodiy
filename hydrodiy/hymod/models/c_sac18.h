
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

/* Number of inputs required by SAC18 run */
#define SAC18_NINPUTS 5

/* Number of params required by SAC18 run */
#define SAC18_NPARAMS 18

/* Number of states returned by SAC18 run */
#define SAC18_NSTATES 10

/* Number of outputs returned by SAC18 run */
#define SAC18_NOUTPUTS 20

int c_sac18_getnstates(void);

int c_sac18_getnoutputs(void);

int sac18_runtimestep(int nparams,
    int nuh,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh,
    double * inputs,
    double * statesuh,
    double * states,
    double * outputs);

int c_sac18_run(int nval,
    int nparams,
    int nuh,
    int ninputs,
    int nstates,
    int noutputs,
    double * params,
    double * uh,
    double * inputs,
    double * statesuhini,
    double * statesini,
    double * outputs);

