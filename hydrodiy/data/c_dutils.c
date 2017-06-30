#include "c_dutils.h"

/**
* Aggregate inputs based on the aggindex
* The operator applied to the aggregation is defined by op:
* oper = 0 : sum
* oper = 1 : mean
**/
int c_aggregate(int nval, int oper, int maxnan, int * aggindex,
    double * inputs, double * outputs, int * iend)
{
    int i, nagg, nagg_nan, count, ia, iaprev;
    double agg, inp, nan;

    /* In case NAN is not defined */
    nan = 0.0/0;

    /* Initialise */
    iaprev = aggindex[0];
    ia = 0;
    count = 0;
    agg = 0;
    nagg = 0;
    nagg_nan = 0;

    for(i=0; i<nval; i++)
    {
        ia = aggindex[i];

        /* Agg index should be increasing */
        if(ia < iaprev)
            return DUTILS_ERROR + __LINE__;

        if(ia != iaprev)
        {
            /* Mean instead of agg */
            if(oper == 1 && nagg>0)
                agg/=nagg;

            /* Store outputs */
            if(nagg_nan > maxnan)
                agg = nan;

            outputs[count] = agg;

            /* Iterates */
            count ++;
            if(count >= nval)
                return DUTILS_ERROR + __LINE__;

            agg = 0;
            nagg = 0;
            nagg_nan = 0;
            iaprev = ia;
        }

        /* check input and skip value if nan */
        inp = inputs[i];

        if(isnan(inp)){
            nagg_nan ++;
            inp = 0;
        } else
            nagg ++;

        agg += inp;
    }

    /* Final step */
    if(oper == 1 && nagg>0)
        agg/=nagg;

    if(nagg_nan > maxnan)
        agg = nan;

    outputs[count] = agg;

    iend[0] = count+1;

    return 0;
}


/*
* code pasted from
* https://stackoverflow.com/questions/24294192/computing-the-binomial-coefficient-in-c
*/
long long c_combi(int n, int k)
{
    long long ans=1;
    int j=1;

    /* Skip if number  is too high */
    if(k>30 || n-k>30){
        return -1;
    }

    k = k>n-k ? n-k : k;

    for(;j<=k;j++,n--)
    {
        if(n%j==0)
        {
            ans *= n/j;
        }else
            if(ans%j==0)
            {
                ans = ans/j*n;
            }else
            {
                ans = (ans*n)/j;
            }
    }
    return ans;
}
