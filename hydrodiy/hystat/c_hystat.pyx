import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_ar1.h':
    int c_ar1random(int nval, double *params,  
            unsigned long int seed, double* output)

cdef extern from 'c_ar1.h':
    int c_ar1innov(int nval, double *params, 
            double * innov, double* output)

cdef extern from 'c_ar1.h':
    int c_ar1inverse(int nval, double *params, 
            double* output, double * innov)

def ar1random(np.ndarray[double, ndim=1, mode='c'] params not None,
        py_seed,
        np.ndarray[double, ndim=1, mode='c'] output not None):
    
    cdef int ierr
    cdef unsigned long int seed=py_seed

    # check dimensions
    assert params.shape[0] == 3

    ierr = c_ar1random(output.shape[0],
            <double*> np.PyArray_DATA(params), seed,
            <double*> np.PyArray_DATA(output))

    return ierr

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

