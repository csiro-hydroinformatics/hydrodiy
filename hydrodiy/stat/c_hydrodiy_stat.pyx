import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_ar1.h':
    int c_ar1innov(int nval, int ncol, double yini, double * alpha,
            double * innov, double* outputs);

cdef extern from 'c_ar1.h':
    int c_ar1inverse(int nval, int ncol, double yini, double * alpha,
            double * inputs, double* innov);

cdef extern from 'c_crps.h':
    int c_crps(int nval,int ncol,
        int use_weights, int is_sorted,
        double* obs,
        double* sim,
        double* weights_vector,
        double* reliability_table,
        double* crps_decompos)

cdef extern from 'c_olsleverage.h':
    int c_olsleverage(int nval, int npreds, double * predictors,
        double * tXXinv, double* leverages)


def __cinit__(self):
    pass


def olsleverage(np.ndarray[double, ndim=2, mode='c'] predictors not None,
        np.ndarray[double, ndim=2, mode='c'] tXXinv not None,
        np.ndarray[double, ndim=1, mode='c'] leverages not None):

    cdef int ierr

    # check dimensions
    assert predictors.shape[0] == leverages.shape[0]
    assert predictors.shape[1] == tXXinv.shape[0]
    assert tXXinv.shape[1] == tXXinv.shape[0]

    ierr = c_olsleverage(predictors.shape[0],
            predictors.shape[1],
            <double*> np.PyArray_DATA(predictors),
            <double*> np.PyArray_DATA(tXXinv),
            <double*> np.PyArray_DATA(leverages))

    return ierr


def ar1innov(double  yini,
        np.ndarray[double, ndim=1, mode='c'] alpha not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert alpha.shape[0] == outputs.shape[0]
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == outputs.shape[1]

    ierr = c_ar1innov(inputs.shape[0], inputs.shape[1], yini,
            <double*> np.PyArray_DATA(alpha),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(outputs))

    return ierr


def ar1inverse(double yini,
        np.ndarray[double, ndim=1, mode='c'] alpha not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=2, mode='c'] innov not None):

    cdef int ierr

    # check dimensions
    assert alpha.shape[0] == innov.shape[0]
    assert inputs.shape[0] == innov.shape[0]
    assert inputs.shape[1] == innov.shape[1]

    ierr = c_ar1inverse(inputs.shape[0], inputs.shape[1], yini,
            <double*> np.PyArray_DATA(alpha),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(innov))

    return ierr


def crps(int use_weights, int is_sorted,
        np.ndarray[double, ndim=1, mode='c'] obs not None,
        np.ndarray[double, ndim=2, mode='c'] sim not None,
        np.ndarray[double, ndim=1, mode='c'] weight_vector not None,
        np.ndarray[double, ndim=2, mode='c'] reliability_table not None,
        np.ndarray[double, ndim=1, mode='c'] crps_decompos not None):

    cdef int ierr

    # check dimensions
    assert obs.shape[0]==sim.shape[0]
    assert crps_decompos.shape[0]==5
    assert reliability_table.shape[0]==sim.shape[1]+1
    assert reliability_table.shape[1]==7

    ierr = c_crps(sim.shape[0], sim.shape[1],
            use_weights, is_sorted,
            <double*> np.PyArray_DATA(obs),
            <double*> np.PyArray_DATA(sim),
            <double*> np.PyArray_DATA(weight_vector),
            <double*> np.PyArray_DATA(reliability_table),
            <double*> np.PyArray_DATA(crps_decompos))

    return ierr


