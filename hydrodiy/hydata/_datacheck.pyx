import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_lindetect.h':
    int c_lindetect(int nval, double* params, 
            double* data, int* linstatus)

def lindetect(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] data not None,
        np.ndarray[int, ndim=1, mode='c'] linstatus not None):
    
    cdef int ierr

    # check dimensions
    assert data.shape[0] == linstatus.shape[0]

    ierr = c_lindetect(data.shape[0], 
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(data),
            <int*> np.PyArray_DATA(linstatus))

    return ierr
