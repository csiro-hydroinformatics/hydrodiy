import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_ar1.h':
    int c_ar1innov(int nval, double *params,
            double * innov, double* output)

cdef extern from 'c_ar1.h':
    int c_ar1inverse(int nval, double *params,
            double* output, double * innov)

cdef extern from 'c_crps.h':
    int c_crps(int nval,int ncol,
        int use_weights, int is_sorted,
        double* obs,
        double* sim,
        double* weights_vector,
        double* reliability_table,
        double* crps_decompos)


def ar1innov(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] innov not None,
        np.ndarray[double, ndim=1, mode='c'] output not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 2
    assert output.shape[0] == innov.shape[0]

    ierr = c_ar1innov(output.shape[0],
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(innov),
            <double*> np.PyArray_DATA(output))

    return ierr


def ar1inverse(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] input not None,
        np.ndarray[double, ndim=1, mode='c'] innov not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 2
    assert input.shape[0] == innov.shape[0]

    ierr = c_ar1inverse(input.shape[0],
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(input),
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


