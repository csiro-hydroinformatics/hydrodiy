#include "c_dateutils.h"



int c_dateutils_isleapyear(int year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int c_dateutils_daysinmonth(int year, int month)
{
    int n;
    int days_in_month[13] = {0, 31, 28, 31, 30, 31, 30,
                    31, 31, 30, 31, 30, 31};

    if(month < 1 || month > 12)
        return -1;

	n = days_in_month[month];

    return c_dateutils_isleapyear(year) == 1 && month == 2 ? n+1 : n;
}


int c_dateutils_dayofyear(int month, int day)
{
    int day_of_year[13] = {0, 0, 31, 59, 90, 120,
                        151, 181, 212, 243, 273, 304, 334};

    if(month < 1 || month > 12)
        return -1;

    if(day < 1 || day > 31)
        return -1;

    /* No need to take leap years into account. This confuses other algorithms */

    return day_of_year[month] + day;
}


int c_dateutils_add1month(int * date)
{
    int nbday;

    /* change month */
    if(date[1] < 12)
    {
        date[1] += 1;
    }
    else
    {
        /* change year */
        date[1] = 1;
        date[0] += 1;
    }

    /* Check that day is not greater than
     * number of days in month */
    nbday = c_dateutils_daysinmonth(date[0], date[1]);
    if(nbday < 0)
        return DATEUTILS_ERROR + __LINE__;

    if(date[2] > nbday)
        date[2] = nbday;

   return 0;
}

int c_dateutils_add1day(int * date)
{
    int nbday;
    nbday = c_dateutils_daysinmonth(date[0], date[1]);

    if(nbday < 0)
        return DATEUTILS_ERROR + __LINE__;

    if(date[2] < nbday)
    {
        date[2] += 1;
        return 0;
    }
    else if(date[2] == nbday) {
        /* change month */
        date[2] = 1;

        if(date[1] < 12)
        {
            date[1] += 1;
            return 0;
        }
        else
        {
            /* change year */
            date[1] = 1;
            date[0] += 1;
            return 0;
        }
    }
    else {
        return DATEUTILS_ERROR + __LINE__;
    }

    return 0;
}

int c_dateutils_getdate(double day, int * date)
{
    int year, month, nday, nbday;

    year = (int)(day * 1e-4);
    month = (int)(day * 1e-2) - year * 100;
    nday = (int)(day) - year * 10000 - month * 100;

    if(month < 0 || month > 12)
        return DATEUTILS_ERROR + __LINE__;

    nbday = c_dateutils_daysinmonth(year, month);
    if(nday < 0 || nday > nbday)
        return DATEUTILS_ERROR + __LINE__;

    date[0] = year;
    date[1] = month;
    date[2] = nday;

    return 0;
}


int c_dateutils_comparedates(int * date1, int * date2)
{
    /* Compare two date values:
    *   1 : date1 < date 2
    *   0 : date1 == date2
    *   -1 : date1 > date2
    */

    if(date1[0]<date2[0])
        return 1;

    if(date1[0]>date2[0])
        return -1;

    if(date1[1]<date2[1])
        return 1;

    if(date1[1]>date2[1])
        return -1;

    if(date1[2]<date2[2])
        return 1;

    if(date1[2]>date2[2])
        return -1;

    return 0;
}
