#include "c_dutils.h"

/**
* Aggregate inputs based on the aggindex
* The operatorator applied to the aggregation is defined by op:
* operator = 0 : sum
* operator = 1 : mean
* operator = 2 : max
* operator = 3 : tail
**/
int c_aggregate(int nval, int operator, int maxnan, int * aggindex,
    double * inputs, double * outputs, int * iend)
{
    int i, nagg, nagg_nan, count, ia, iaprev;
    double agg, inp, nan;
    static double zero = 0.0;

    /* In case NAN is not defined */
    nan = 1./zero * zero;

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
            if(operator == 1 && nagg>0)
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

        if(isnan(inp))
        {
            nagg_nan ++;
            inp = 0;
        } else
            nagg ++;

        if(operator<=1) {
            agg += inp;
        }
        else if (operator == 2){
            agg = max(agg, inp);
        }
        else if (operator == 3){
            agg = inp;
        }
    }

    /* Final step */
    if(operator == 1 && nagg>0)
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


/**
* Convert a time series into a flat homogeneised time series
**/
int c_flathomogen(int nval, int maxnan, int * aggindex,
    double * inputs, double * outputs)
{
    int i, j, nagg, nagg_nan, start, ia, iaprev;
    double agg, inp, nan;
    static double zero = 0.0;

    /* In case NAN is not defined */
    nan = zero/zero;

    /* Initialise */
    iaprev = aggindex[0];
    ia = 0;
    start = 0;
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
            /* Set agg to nan if too many nans */
            if(nagg_nan > maxnan)
                agg = nan;

            /* Set outputs value to flat homogeised ones */
            for(j=start; j<i; j++)
            {
                inp = inputs[j];
                if(isnan(inp))
                {
                    outputs[j] = nan;
                } else
                {
                    outputs[j] = agg/nagg;
                }
            }
            /* Iterates */
            agg = 0;
            nagg = 0;
            nagg_nan = 0;
            iaprev = ia;
            start = i;
        }

        /* check input and skip value if nan */
        inp = inputs[i];
        if(isnan(inp))
        {
            nagg_nan ++;
            inp = 0;
        } else
            nagg ++;

        agg += inp;
    }


    /* Final step */
    if(nagg_nan > maxnan)
        agg = nan;

    for(j=start; j<i; j++)
    {
        inp = inputs[j];
        if(isnan(inp))
        {
            outputs[j] = nan;
        } else
        {
            outputs[j] = agg/nagg;
        }
    }

   return 0;
}

