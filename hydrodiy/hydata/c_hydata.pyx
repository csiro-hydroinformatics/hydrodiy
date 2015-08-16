import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_baseflow.h':
    int c_baseflow(int method, int nval, double* params, 
            double* inputs, double* outputs)

def baseflow(int method, 
		np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] outputs not None):
    
    cdef int ierr

    # check dimensions
    assert inputs.shape[0] == outputs.shape[0]

    ierr = c_baseflow(method,
			inputs.shape[0], 
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(outputs))

    return ierr
