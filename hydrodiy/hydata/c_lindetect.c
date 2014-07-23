#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>


/* ************** Core subroutine ******************************
* Detect the presence of linear interpolation in data series
* nval = length of input vectors (number of values)
* params = algorithm parameters
*   params[0] = number of points (good idea ~ 3)
*   params[1] = tolerance (good idea ~ 1e-5)
* data = data series
* linstatus = linearly interpolated (1) or not (0)
*
************************************************/
int c_lindetect(int nval, double *params, double* data, int* linstatus)
{
	int i, npt;
	double v1, v2, err, interp, tol;

    /* Get parameters */
    npt = (int)params[0];
    tol = params[1];

    /* Check inputs */
    if((double)npt>(double)(nval/10)){
        return EDOM;
    }
    if((tol<0) | (npt<1)){
        return EDOM;
    }

    /* fill up beginning and end of linstatus vector */
    for(i=0; i<npt; i++) linstatus[i] = 0;
    for(i=nval-npt; i<nval; i++) linstatus[i] = 0;

    /* loop through data */
    for(i=npt; i<nval-npt; i++){
            v1 = data[i-npt];
            v2 = data[i+npt];
            interp = (v2-v1)/(2*(double)npt)*(double)npt+v1;
            linstatus[i] = 0;
            err = fabs(data[i]-interp)/(1+fabs(data[i]+interp)/2);
            if((err<tol) & ((v1!=0)|(v2!=0))) linstatus[i] = 1;
    }

    return 0;
}
