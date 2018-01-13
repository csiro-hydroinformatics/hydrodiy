#include <stdio.h>
#include <math.h>
#include <errno.h>

/* Define Error message */
#define ANDARL_ERROR 500000

/* Function */
double adinf(double z);

double errfix(int n,double x);

double AD(int n,double z);

int ADtest(int n, double *x, double *outputs);

