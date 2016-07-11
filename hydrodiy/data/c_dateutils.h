
#ifndef __DATEUTILS__
#define __DATEUTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Define Error message */
#define DATEUTILS_ERROR 100000

/* utility functions */

int c_dateutils_isleapyear(int year);


int c_dateutils_daysinmonth(int year, int month);


int c_dateutils_dayofyear(int month, int day);


int c_dateutils_add1month(int * date);


int c_dateutils_add1day(int * date);


int c_dateutils_comparedates(int * date1, int * date2);


int c_dateutils_getdate(double day, int * date);

#endif
