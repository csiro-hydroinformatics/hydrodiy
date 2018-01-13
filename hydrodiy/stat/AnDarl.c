/*
 * This code is part of the paper by Marsaglia and Marsaglia (2004)
 * Marsaglia, G., & Marsaglia, J. (2004). Evaluating the Anderson-Darling Distribution.
 * Journal of Statistical Software, 9(2), 1 - 5. doi:http://dx.doi.org/10.18637/jss.v009.i02
 *
 * File downloaded from https://www.jstatsoft.org/article/view/v009i02
 * Code was reformatted to add clarity. Main subroutine was also removed.
 *
 *
 * */

#include "AnDarl.h"

/*
    Anderson-Darling test for uniformity.   Given an ordered set
              x_1<x_2<...<x_n
 of purported uniform [0,1) variates,  compute
          a = -n-(1/n)*[ln(x_1*z_1)+3*ln(x_2*z_2+...+(2*n-1)*ln(x_n*z_n)]
 where z_1=1-x_n, z_2=1-x_(n-1)...z_n=1-x_1, then find
  v=adinf(a) and return  p=v+errfix(v), which should be uniform in [0,1),
  that is, the p-value associated with the observed x_1<x_2<...<x_n.
*/

/* Short, practical version of full ADinf(z), z>0.   */
double adinf(double z)
{
    if(z<2.)
        return exp(-1.2337141/z)/sqrt(z)*(2.00012+(.247105-  \
                        (.0649821-(.0347962-(.011672-.00168691*z)*z)*z)*z)*z);

    /* max |error| < .000002 for z<2, (p=.90816...) */

    return exp(-exp(1.0776-(2.30695-(.43424-(.082433-(.008056 -.0003146*z)*z)*z)*z)*z));
    /* max |error|<.0000008 for 4<z<infinity */
}

/*
The procedure  errfix(n,x)  corrects the error caused
by using the asymptotic approximation, x=adinf(z).
Thus x+errfix(n,x) is uniform in [0,1) for practical purposes;
accuracy may be off at the 5th, rarely at the 4th, digit.
*/
double errfix(int n, double x)
{
    double c,t;

    if(x>.8)
        return (-130.2137+(745.2337-(1705.091-(1950.646-(1116.360-255.7844*x)*x)*x)*x)*x)/n;

    c=.01265+.1757/n;
    if(x<c)
    {
        t=x/c;
        t=sqrt(t)*(1.-t)*(49*t-102);
        return t*(.0037/(n*n)+.00078/n+.00006)/n;
	}

    t=(x-c)/(.8-c);
    t=-.00022633+(6.54034-(14.6538-(14.458-(8.259-1.91864*t)*t)*t)*t)*t;
    return t*(.04213+.01365/n)/n;
}

/* The function AD(n,z) returns Prob(A_n<z) where
    A_n = -n-(1/n)*[ln(x_1*z_1)+3*ln(x_2*z_2+...+(2*n-1)*ln(x_n*z_n)]
          z_1=1-x_n, z_2=1-x_(n-1)...z_n=1-x_1, and
    x_1<x_2<...<x_n is an ordered set of iid uniform [0,1) variates.
*/

double AD(int n,double z)
{
    double c,v,x;
    x=adinf(z);

    /* now x=adinf(z). Next, get v=errfix(n,x) and return x+v; */
	if(x>.8)
    {
        v=(-130.2137+(745.2337-(1705.091-(1950.646-(1116.360-255.7844*x)*x)*x)*x)*x)/n;
        return x+v;
    }

    c=.01265+.1757/n;
    if(x<c)
    {
        v=x/c;
        v=sqrt(v)*(1.-v)*(49*v-102);
        return x+v*(.0037/(n*n)+.00078/n+.00006)/n;
    }

    v=(x-c)/(.8-c);
    v=-.00022633+(6.54034-(14.6538-(14.458-(8.259-1.91864*v)*v)*v)*v)*v;
    return x+v*(.04213+.01365/n)/n;
}

/* You must give the ADtest(int n, double *x) routine a sorted array
       x[0]<=x[1]<=..<=x[n-1]
    that you are testing for uniformity.
   It will return the p-value associated
   with the Anderson-Darling test, using
    the above adinf() and errfix( ,   )
         Not well-suited for n<7,
     (accuracy could drop to 3 digits).

    Function has been modified to return the
    test statistic and the pvalue.

    The function also aborts if data is not
    sorted, or if data is outside the
    range [0, 1]
*/

int ADtest(int n, double *x, double *outputs)
{
    int i;
    double nan, t,z=0, prev=-1e-300;
    static double zero = 0.0;

    /* nan value if not defined */
    nan = zero/zero;
    outputs[0] = nan;
    outputs[1] = nan;

    /* compute AD statistic */
    for(i=0;i<n;i++)
    {
        /* Check input data */
        if(x[i]<0 || x[i]>1)
            return ANDARL_ERROR + __LINE__;

        if(isnan(x[i]))
            return ANDARL_ERROR + __LINE__;

        if(x[i]<prev)
            return ANDARL_ERROR + __LINE__;

        /* Compute test stat */
        t=x[i]*(1.-x[n-1-i]);
        z=z-(i+i+1)*log(t);

        prev = x[i];
    }

    /* Store outputs */
    outputs[0] = -n+z/n;
    outputs[1] = 1.-AD(n, -n+z/n);

    return 0;
 }


