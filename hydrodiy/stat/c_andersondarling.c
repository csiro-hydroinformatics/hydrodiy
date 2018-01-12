#include "c_andersondarling.h"

int c_ad_probexactinf(int nval, double *data, double *prob) {
  int i;
  for(i = 0; i < nval; i++)
    prob[i] = ADinf(data[i]);

  return 0;
}

int c_ad_probn(int nval, int nsample, double *data, double *prob) {
  int i;
  for(i = 0; i < nval; i++)
    prob[i] = AD(nsample, data[i]);

  return 0;
}

int c_ad_probapproxinf(int nval, double *data, double *prob) {
  int i;
  for(i = 0; i < nval; i++)
    prob[i] = adinf(data[i]);

  return 0;
}

